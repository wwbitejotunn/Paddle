// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cub/cub.cuh>
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/cutlass_forward.h"

namespace phi {
namespace fusion {

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return max(a, b);
  }
};

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int *seq_lens,
                                int *max_len,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    max_len_this_thread = max(seq_lens[i], max_len_this_thread);
  }
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  if (tid == 0) {
    *max_len = total;
  }
}

int GetMaxLen(const phi::GPUContext &ctx,
              const phi::DenseTensor &seq_lens_tensor,
              phi::DenseTensor *max_len_tensor,
              const int batch_size) {
  constexpr int blockSize = 128;
  int max_len_cpu = 0;
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, ctx.stream()>>>(
      seq_lens_tensor.data<int>(), max_len_tensor->data<int>(), batch_size);
  phi::memory_utils::Copy(phi::CPUPlace(),
                          &max_len_cpu,
                          ctx.GetPlace(),
                          max_len_tensor->data<int>(),
                          sizeof(int),
                          ctx.stream());
  return max_len_cpu;
}

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(T *q_buf,
                                                T *kv_buf,
                                                const T *qkv,
                                                const int *padding_offset,
                                                const int *seq_lens,
                                                const int32_t elem_cnt,
                                                const int batch_size,
                                                const int max_len_this_time,
                                                const int seq_len,
                                                const int token_num,
                                                const int head_num,
                                                const int size_per_head) {
  const int32_t offset =
      batch_size * max_len_this_time * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {
      phi::Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * max_len_this_time *
                     size_per_head +
                 head_id * max_len_this_time * size_per_head +
                 seq_id * size_per_head + size_id]);
    } else {
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Store<T, VecSize>(
          src_vec,
          &kv_buf[kv_store_offset +
                  target_batch_id * head_num * max_len_this_time *
                      size_per_head +
                  head_id * max_len_this_time * size_per_head +
                  seq_id * size_per_head + size_id]);
    }
  }
}

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMultiProcessors(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return cudaSuccess;
}

constexpr int VEC_16B = 16;

template <typename T>
void qkv_transpose_split(const phi::GPUContext &ctx,
                         T *q_buf,
                         T *kv_buf,
                         const T *qkv,
                         const int *padding_offset,
                         const int *seq_lens,
                         const int token_num,
                         const int batch_size,
                         const int head_num,
                         const int max_len_this_time,
                         const int seq_len,
                         const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    phi::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, ctx.stream()>>>(q_buf,
                                                  kv_buf,
                                                  qkv,
                                                  padding_offset,
                                                  seq_lens,
                                                  elem_cnt,
                                                  batch_size,
                                                  max_len_this_time,
                                                  seq_len,
                                                  token_num,
                                                  head_num,
                                                  size_per_head);
}

template <typename T, int VecSize>
__global__ void TransposeRemovingPadding(const T *input_data,
                                         const int *seq_lens,
                                         T *output_data,
                                         const int batch_size,
                                         const int num_head,
                                         const int max_len_this_time,
                                         const int seq_len,
                                         const int head_dim,
                                         const int token_num,
                                         const int elem_cnt,
                                         const int *padding_offset) {
  // transpose and remove padding
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num,
  // num_head, head_dim]
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int dim_embed = num_head * head_dim;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / dim_embed;
    const int ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int ori_batch_id = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_batch_id] == 0) continue;
    const int ori_seq_id = ori_token_idx % seq_len;
    const int ori_head_id = (linear_index % dim_embed) / head_dim;
    const int ori_head_lane = (linear_index % dim_embed) % head_dim;
    const int ori_idx = ori_batch_id * num_head * max_len_this_time * head_dim +
                        ori_head_id * max_len_this_time * head_dim +
                        ori_seq_id * head_dim + ori_head_lane;
    phi::Load<T, VecSize>(&input_data[ori_idx], &src_vec);
    phi::Store<T, VecSize>(src_vec, &output_data[linear_index]);
  }
}

template <typename T>
void InvokeTransposeRemovePadding(const phi::GPUContext &ctx,
                                  const T *input_data,
                                  const int *seq_lens,
                                  T *output_data,
                                  const int batch_size,
                                  const int num_head,
                                  const int max_len_this_time,
                                  const int seq_len,
                                  const int head_dim,
                                  const int token_num,
                                  const int *padding_offset) {
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num,
  // num_head, head_dim]
  constexpr int VEC_16B = 16;
  const int elem_cnt = token_num * num_head * head_dim;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      head_dim % PackSize,
      0,
      phi::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", head_dim, PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t block_size = 128;
  int32_t grid_size = (pack_num + block_size - 1) / block_size;
  TransposeRemovingPadding<T, PackSize>
      <<<grid_size, block_size, 0, ctx.stream()>>>(input_data,
                                                   seq_lens,
                                                   output_data,
                                                   batch_size,
                                                   num_head,
                                                   max_len_this_time,
                                                   seq_len,
                                                   head_dim,
                                                   token_num,
                                                   elem_cnt,
                                                   padding_offset);
}

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context &ctx,
    const DenseTensor &qkv,  // [token_num, num_head, dim_head]
    const DenseTensor &seq_lens,
    const DenseTensor &padding_offset,  // [token_num]
    const paddle::optional<DenseTensor>
        &pre_cache,  // [2, bs, num_head, prompt_num, dim_head]
    const paddle::optional<DenseTensor> &mask,
    const float scale,
    const bool causal,
    DenseTensor *output) {
  ctx.template Alloc<T>(output);
  auto qkv_dims = qkv.dims();
  const int64_t token_num = qkv_dims[0];
  const int64_t num_head = qkv_dims[2];
  const int64_t dim_head = qkv_dims[3];
  const int64_t bsz = seq_lens.dims()[0];

  phi::DenseTensor max_len_tensor;
  max_len_tensor.Resize({{1}});
  auto *max_len_data = ctx.template Alloc<int>(
      &max_len_tensor, max_len_tensor.numel() * sizeof(int));
  int max_len_this_time = GetMaxLen(ctx, seq_lens, &max_len_tensor, bsz);

  phi::DenseTensor q_transpose_out, kv_transpose_out, tmp_out;
  q_transpose_out.Resize({{bsz, num_head, max_len_this_time, dim_head}});
  auto *q_transpose_out_data = ctx.template Alloc<T>(
      &q_transpose_out, q_transpose_out.numel() * sizeof(T));
  kv_transpose_out.Resize({{2, bsz, num_head, max_len_this_time, dim_head}});
  auto *kv_transpose_out_data = ctx.template Alloc<T>(
      &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));
  tmp_out.Resize({{bsz, num_head, max_len_this_time, dim_head}});
  auto *tmp_out_data =
      ctx.template Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  qkv_transpose_split<T>(ctx,
                         q_transpose_out.data<T>(),
                         kv_transpose_out.data<T>(),
                         qkv.data<T>(),
                         padding_offset.data<int>(),
                         seq_lens.data<int>(),
                         token_num,
                         bsz,
                         num_head,
                         max_len_this_time,
                         max_len_this_time,
                         dim_head);
  // VLOG(0) << "q: " << q_transpose_out;
  // VLOG(0) << "kv: " << kv_transpose_out;
  int prompt_num = 0;
  phi::DenseTensor cache_kv_out_tensor;
  if (pre_cache) {
    auto pre_cache_tensor = pre_cache.get();
    prompt_num = pre_cache_tensor.dims()[3];
    cache_kv_out_tensor.Resize(
        {{2, bsz, num_head, max_len_this_time + prompt_num, dim_head}});
    auto *cache_kv_out_data = ctx.template Alloc<T>(
        &cache_kv_out_tensor, cache_kv_out_tensor.numel() * sizeof(T));
    phi::funcs::ConcatFunctor<Context, T> concat;
    concat(ctx, {pre_cache_tensor, kv_transpose_out}, 3, &cache_kv_out_tensor);
  }

  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens.data<int>();

  params.num_batches = bsz;
  params.num_heads = num_head;
  params.query_seq_len = max_len_this_time;
  params.head_size = dim_head;
  params.key_value_seq_len = max_len_this_time + prompt_num;
  params.value_head_size = dim_head;

  T *kv_data =
      pre_cache ? cache_kv_out_tensor.data<T>() : kv_transpose_out_data;
  params.datatype = q_transpose_out.dtype();
  params.query_ptr = q_transpose_out_data;
  params.key_ptr = kv_data;
  params.value_ptr =
      kv_data + bsz * num_head * params.key_value_seq_len * dim_head;

  params.output_ptr = tmp_out_data;

  params.ldq = params.head_size;
  params.ldk = params.head_size;
  params.ldv = params.value_head_size;
  params.ldo = params.value_head_size;

  params.ElementQ = params.query_seq_len * params.head_size;
  params.ElementK = params.key_value_seq_len * params.head_size;
  params.ElementV = params.key_value_seq_len * params.value_head_size;
  params.ElementO = params.query_seq_len * params.value_head_size;

  params.scale = scale;
  params.causal = causal;

  if (mask) {
    // [B, 1, S, D]
    auto mask_tensor = mask.get();
    params.ldm = mask_tensor.dims()[3];
    params.ElementM = mask_tensor.dims()[2] * mask_tensor.dims()[3];
    params.mask_ptr = mask_tensor.data();
    params.mask_broadcast_row = false;
  } else {
    params.mask_ptr = nullptr;
    params.mask_broadcast_row = false;
  }

  bool kernel_launched = false;

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    using KernelType = decltype(k_);
    if (kernel_launched) {
      return;
    }
    if (mask && !KernelType::kAddMask) {
      return;
    }
    if (!mask && KernelType::kAddMask) {
      return;
    }
    if (KernelType::kMaskBroadcastRow) {
      // not support mask_broad_cast
      return;
    }
    if (params.mask_ptr &&
        reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
        params.ldm % (16 / sizeof(T)) == 0 && !KernelType::kMaskIsAligned) {
      return;
    }
    if (params.mask_ptr &&
        !(reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
          params.ldm % (16 / sizeof(T)) == 0) &&
        KernelType::kMaskIsAligned) {
      return;
    }
    if (KernelType::kSingleValueIteration &&
        KernelType::kKeysPerBlock < params.value_head_size) {
      return;
    }
    if (KernelType::kKeysPerBlock == 64 && params.value_head_size > 64) {
      return;
    }
    if (params.head_size % KernelType::MM0::kAlignmentA) {
      return;
    }
    kernel_launched = true;
    kernel_fn(k_, params, ctx);
  };
  dispatch_cutlass_forward<T, decltype(launchKernel)>(ctx, launchKernel);
  PADDLE_ENFORCE_EQ(
      kernel_launched,
      true,
      phi::errors::InvalidArgument("the kernel should not be launched"));
  InvokeTransposeRemovePadding<T>(ctx,
                                  tmp_out_data,
                                  seq_lens.data<int>(),
                                  output->data<T>(),
                                  bsz,
                                  num_head,
                                  max_len_this_time,
                                  max_len_this_time,
                                  dim_head,
                                  token_num,
                                  padding_offset.data<int>());
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(memory_efficient_attention_variable,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiHeadAttentionVariableForwardKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  // kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}

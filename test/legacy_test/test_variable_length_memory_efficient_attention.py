# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import unittest

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid import Program, core, program_guard
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,
)

paddle.seed(2023)


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def create_attn_mask(
    mask_type,
    batch_size,
    seq_lens,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros(
        [batch_size, 1, max_seq_len, max_seq_len], dtype=mask_type
    )
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type))
            - 1
        ) * 1e4
    return mask


def naive_attention_impl(query, key, value, mask, scale):
    q = query.transpose([0, 2, 1, 3])
    k = key.transpose([0, 2, 1, 3])
    v = value.transpose([0, 2, 1, 3])
    qk_res = paddle.matmul(q, k, transpose_y=True)
    attention = qk_res * scale
    attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(softmax_result, v).transpose([0, 2, 1, 3])
    return result


def get_padding_offset(seq_lens, max_seq_len, batch_size):
    token_num = paddle.sum(seq_lens)
    padding_offset = paddle.zeros([token_num], "int32")
    index = 0
    cum_offset = 0
    for i in range(batch_size):
        for j in range(seq_lens[i]):
            padding_offset[index] = cum_offset
            index += 1
        cum_offset += max_seq_len - seq_lens[i]
    return padding_offset


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.2",
)
class TestMemEffAttentionVariableAPI(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAPIVariable_fp32"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 64
        self.token_num = self.batch_size * self.seq_len
        self.dim_head = 16
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.shape = (
            self.token_num,
            3,
            self.num_head,
            self.dim_head,
        )
        self.dtype = 'float32'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])

    def test_all(self):
        print(
            f"Test All case shape {self.shape} dtype {self.dtype} name {self.name}"
        )

        paddle.disable_static()

        qkv = np.random.random(self.shape)
        qkv_tensor = paddle.to_tensor(qkv, self.dtype)

        query = qkv_tensor[:, 0, :, :].reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        key = qkv_tensor[:, 1, :, :].reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )
        value = qkv_tensor[:, 2, :, :].reshape(
            [self.batch_size, self.seq_len, self.num_head, self.dim_head]
        )

        out_ = naive_attention_impl(
            query, key, value, self.attention_mask, self.scale
        ).reshape((-1, self.num_head, self.dim_head))

        padding_offset = get_padding_offset(
            self.seq_lens, paddle.max(self.seq_lens), self.batch_size
        )
        out = variable_length_memory_efficient_attention(
            qkv_tensor,
            self.seq_lens,
            padding_offset,
            mask=self.attention_mask,
            scale=self.scale,
        )
        out_casual = variable_length_memory_efficient_attention(
            qkv_tensor,
            self.seq_lens,
            padding_offset,
            scale=self.scale,
            causal=True,
        )

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)
        np.testing.assert_allclose(
            out_casual.numpy(), out_, rtol=5e-03, atol=1e-03
        )


class TestMemEffAPIVariableDtypeFP16(TestMemEffAttentionVariableAPI):
    def setUp(self):
        self.name = "MemEffAPIVariable_fp16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 3
        self.num_head = 16
        self.seq_len = 64
        self.token_num = self.batch_size * self.seq_len
        self.dim_head = 64
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.shape = (
            self.token_num,
            3,
            self.num_head,
            self.dim_head,
        )
        self.dtype = 'float16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])


class TestMemEffAPIVariableDtypeBF16(TestMemEffAttentionVariableAPI):
    def setUp(self):
        self.name = "MemEffAPIVariable_bf16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 1
        self.num_head = 8
        self.seq_len = 32
        self.token_num = self.batch_size * self.seq_len
        self.dim_head = 128
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.shape = (
            self.token_num,
            3,
            self.num_head,
            self.dim_head,
        )
        self.dtype = 'bfloat16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11020,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.2",
)
class TestMemEffAPIVariableDtypeFP16Static(unittest.TestCase):
    def setUp(self):
        self.name = "MemEffAPIVariableStatic_fp16"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 3
        self.num_head = 16
        self.seq_len = 64
        self.token_num = self.batch_size * self.seq_len
        self.dim_head = 32
        self.seq_lens = paddle.to_tensor(
            [
                self.seq_len,
            ]
            * self.batch_size,
            "int32",
        )
        self.padding_offset = get_padding_offset(
            self.seq_lens, paddle.max(self.seq_lens), self.batch_size
        ).numpy()
        self.seq_lens = self.seq_lens.numpy()
        self.shape = (
            self.token_num,
            3,
            self.num_head,
            self.dim_head,
        )
        self.dtype = 'float16'
        self.attention_mask = create_attn_mask(
            self.dtype,
            self.batch_size,
            [
                self.seq_len,
            ]
            * self.batch_size,
        ).numpy()
        self.qkv = np.random.random(self.shape).astype(self.dtype)
        self.q = (
            self.qkv[:, 0, :, :]
            .reshape(
                [self.batch_size, self.seq_len, self.num_head, self.dim_head]
            )
            .astype(self.dtype)
        )
        self.k = (
            self.qkv[:, 1, :, :]
            .reshape(
                [self.batch_size, self.seq_len, self.num_head, self.dim_head]
            )
            .astype(self.dtype)
        )
        self.v = (
            self.qkv[:, 2, :, :]
            .reshape(
                [self.batch_size, self.seq_len, self.num_head, self.dim_head]
            )
            .astype(self.dtype)
        )
        self.scale = 1.0 / np.sqrt(self.shape[-1])

        self.ref_out = naive_attention_impl(
            paddle.to_tensor(self.q),
            paddle.to_tensor(self.k),
            paddle.to_tensor(self.v),
            paddle.to_tensor(self.attention_mask),
            self.scale,
        ).reshape([-1, self.num_head, self.dim_head])

    def test_all(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            qkv = paddle.static.data(
                name="qkv", shape=self.shape, dtype=self.dtype
            )
            seq_lens = paddle.static.data(
                name="seq_lens", shape=[self.batch_size, 1], dtype="int32"
            )
            padding_offset = paddle.static.data(
                name="padding_offset",
                shape=[self.token_num],
                dtype="int32",
            )
            mask = paddle.static.data(
                name="mask",
                shape=[self.batch_size, 1, self.seq_len, self.seq_len],
                dtype=self.dtype,
            )
            out = variable_length_memory_efficient_attention(
                qkv, seq_lens, padding_offset, mask=mask, scale=self.scale
            )
            exe = fluid.Executor(paddle.CUDAPlace(0))
            res = exe.run(
                feed={
                    "qkv": self.qkv,
                    "seq_lens": self.seq_lens,
                    "padding_offset": self.padding_offset,
                    "mask": self.attention_mask,
                },
                fetch_list=[out],
            )
        paddle.disable_static()
        np.testing.assert_allclose(res[0], self.ref_out, rtol=5e-03, atol=1e-03)


if __name__ == '__main__':
    unittest.main()

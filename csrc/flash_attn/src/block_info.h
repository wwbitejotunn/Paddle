/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : (params.chunked_seq_lens_q == nullptr ? params.cu_seqlens_q[bidb + 1] - sum_s_q : params.chunked_seq_lens_q[bidb]))
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? (params.chunked_seq_lens_k == nullptr ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.chunked_seq_lens_k[bidb]) : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        , chunked_seq_continus_token_lens_k(params.chunked_seq_continus_token_lens_k? params.chunked_seq_continus_token_lens_k[bidb]:0)
        , chunked_seq_sink_token_mask_lens_k(params.chunked_seq_sink_token_mask_lens_k? params.chunked_seq_sink_token_mask_lens_k[bidb]:0)
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
    const int chunked_seq_continus_token_lens_k;
    const int chunked_seq_sink_token_mask_lens_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash

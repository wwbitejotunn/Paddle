# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# The following codes are from https://github.com/facebookresearch/xformers

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

from paddle import _C_ops
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def variable_length_memory_efficient_attention(
    qkv,
    seq_lens,
    padding_offset,
    pre_cache=None,
    mask=None,
    scale=None,
    causal=False,
):
    """
    Cutlass Memory Efficient Variable Attention.
    This method requires SM_ARCH in sm70, sm75, sm80.

    Args:
        qkv (Tensor): the Query, Key, Value Tensor. Its shape is [token_num, 3, num_head, head_size].
        seq_lens (Tensor): the SequenceLengths Tensor. Its shape is [batchsize, 1].
        padding_offset (Tensor): the padding offset Tensor. Its shape is [token_num].
        mask (Tensor): the Mask Tensor. Its shape is [batchsize, 1, query_seq_len, key_seq_len].
        pre_cahce (Tensor): the pre-cache Tensor. Its shape is [2, batchsize, num_heads, seq_len, head_size]
        scale (Float): the attention matrix's scale. Default is sqrt(1.0 / head_size).
        causal (Bool): whether causal masking is used or not. Default is False.
    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import math
            import paddle
            from paddle.incubate.nn.functional import variable_length_memory_efficient_attention

            def get_padding_offset(seq_lens, max_seq_len, batch_size):
                token_num = paddle.sum(seq_lens)
                padding_offset = paddle.zeros([token_num], "int32")
                index = 0
                cum_offset = 0
                for i in range(batch_size):
                    for j in range(seq_lens[i]):
                        padding_offset[index] = cum_offset
                        index += 1
                    cum_offset += (max_seq_len - seq_lens[i])
                return padding_offset

            batch = 1
            num_head = 8
            seq_len = 256
            token_num = batch * seq_len
            head_size = 32

            dtype = paddle.float16

            qkv = paddle.randn([token_num, 3, num_head, head_size], dtype)
            query = qkv[:, 0, :, :].reshape([batch, seq_len, num_head, head_size])
            key = qkv[:, 1, :, :].reshape([batch, seq_len, num_head, head_size])
            value = qkv[:, 2, :, :].reshape([batch, seq_len, num_head, head_size])
            seq_lens = paddle.to_tensor([seq_len, ] * batch, dtype='int32')
            padding_offset = get_padding_offset(seq_lens, paddle.max(seq_lens), batch)
            mask = paddle.randn([batch, 1, seq_len, seq_len], dtype=dtype)

            scale = float(1.0 / math.sqrt(head_size))

            def naive_attention_impl(query, key, value, mask, scale):
                query = paddle.transpose(query, [0, 2, 1, 3])
                key = paddle.transpose(key, [0, 2, 1, 3])
                value = paddle.transpose(value, [0, 2, 1, 3])

                qk_res = paddle.matmul(query, key, transpose_y=True)
                attention = qk_res * scale
                attention = attention + mask
                softmax_result = paddle.nn.functional.softmax(attention, -1)
                result = paddle.matmul(softmax_result, value)
                result = paddle.transpose(result, [0, 2, 1, 3]).reshape([-1, num_head, head_size])
                return result

            out = naive_attention_impl(query, key, value, mask, scale)
            # equals to: out = variable_length_memory_efficient_attention(qkv, seq_lens, padding_offset, mask=mask, scale=scale)

            print(out.shape) # [token_num, num_head, head_size]
    """
    if scale is None:
        head_size = qkv.shape[3]
        scale = float(1.0 / math.sqrt(head_size))

    if in_dynamic_mode():
        return _C_ops.variable_length_memory_efficient_attention(
            qkv, seq_lens, padding_offset, pre_cache, mask, scale, causal
        )

    helper = LayerHelper(
        'variable_length_memory_efficient_attention', **locals()
    )
    out = helper.create_variable_for_type_inference(dtype=qkv.dtype)
    helper.append_op(
        type='variable_length_memory_efficient_attention',
        inputs={
            'qkv': qkv,
            'seq_lens': seq_lens,
            'padding_offset': padding_offset,
            'pre_cache': pre_cache,
            "mask": mask,
        },
        attrs={"scale": scale, "causal": causal},
        outputs={'out': out},
    )
    return out

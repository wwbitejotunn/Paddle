# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertTransLayernormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def conv_filter_datagen(dics):
            x = np.random.randn(dics["c"], dics["c"], 1, 1) * 0.01
            return x.astype(np.float32)

        def elementwise_bias_datagen(dics):
            x = np.random.randn(dics["c"]) * 0.01
            return x.astype(np.float32)

        def layernorm_bias_datagen(dics):
            x = np.random.randn(dics["c"]) * 0.01
            return x.astype(np.float32)

        def layernorm_scale_datagen(dics):
            x = np.ones([dics["c"]])
            return x.astype(np.float32)

        def conv2d_input_datagen(dics):
            x = np.random.randn(dics["batch"], dics["c"], dics["h"], dics["w"])
            return x.astype(np.float32)

        for batch in [2]:
            for begin_norm_axis in [2]:
                for h in [32, 64]:
                    for w in [32, 64]:
                        for c in [128, 320, 255]:
                            for reshape in ["flatten", "reshape"]:
                                dics = {
                                    "batch": batch,
                                    "begin_norm_axis": begin_norm_axis,
                                    "h": h,
                                    "w": w,
                                    "c": c,
                                    "flatten": {
                                        "op_type": "flatten_contiguous_range",
                                        "op_inputs": {
                                            "X": ["transpose2_out"],
                                        },
                                        "op_outputs": {
                                            "Out": ["reshape_out"],
                                        },
                                        "op_attrs": {
                                            "start_axis": 1,
                                            "stop_axis": 2,
                                        },
                                    },
                                    "reshape": {
                                        "op_type": "reshape2",
                                        "op_inputs": {
                                            "X": ["transpose2_out"],
                                        },
                                        "op_outputs": {
                                            "Out": ["reshape_out"],
                                        },
                                        "op_attrs": {"shape": [-1, h * w, c]},
                                    },
                                }
                                ops_config = [
                                    {
                                        "op_type": "conv2d",
                                        "op_inputs": {
                                            "Input": ["conv2d_input"],
                                            "Filter": ["conv2d_filter"],
                                        },
                                        "op_outputs": {
                                            "Output": ["conv2d_output"],
                                        },
                                        "op_attrs": {
                                            "dilations": [1, 1],
                                            "padding_algorithm": "EXPLICIT",
                                            "groups": 1,
                                            "paddings": [0, 0],
                                            "strides": [1, 1],
                                        },
                                    },
                                    {
                                        "op_type": "elementwise_add",
                                        "op_inputs": {
                                            "X": ["conv2d_output"],
                                            "Y": ["elementwise_bias"],
                                        },
                                        "op_outputs": {
                                            "Out": ["elementwise_out"]
                                        },
                                        "op_attrs": {"axis": 1},
                                    },
                                    {
                                        "op_type": "transpose2",
                                        "op_inputs": {
                                            "X": ["elementwise_out"],
                                        },
                                        "op_outputs": {
                                            "Out": ["transpose2_out"],
                                        },
                                        "op_attrs": {"axis": [0, 2, 3, 1]},
                                    },
                                    dics[reshape],
                                    {
                                        "op_type": "layer_norm",
                                        "op_inputs": {
                                            "X": ["reshape_out"],
                                            "Bias": ["layernorm_bias"],
                                            "Scale": ["layernorm_scale"],
                                        },
                                        "op_outputs": {
                                            "Y": ["layernorm_out"],
                                            "Mean": ["layernorm_mean"],
                                            "Variance": ["layernorm_variance"],
                                        },
                                        "op_attrs": {
                                            "epsilon": 1e-5,
                                            "begin_norm_axis": dics[
                                                "begin_norm_axis"
                                            ],
                                        },
                                    },
                                ]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={
                                        "conv2d_filter": TensorConfig(
                                            data_gen=partial(
                                                conv_filter_datagen, dics
                                            )
                                        ),
                                        "elementwise_bias": TensorConfig(
                                            data_gen=partial(
                                                elementwise_bias_datagen, dics
                                            )
                                        ),
                                        "layernorm_bias": TensorConfig(
                                            data_gen=partial(
                                                layernorm_bias_datagen, dics
                                            )
                                        ),
                                        "layernorm_scale": TensorConfig(
                                            data_gen=partial(
                                                layernorm_scale_datagen, dics
                                            )
                                        ),
                                    },
                                    inputs={
                                        "conv2d_input": TensorConfig(
                                            data_gen=partial(
                                                conv2d_input_datagen, dics
                                            )
                                        ),
                                    },
                                    outputs=["reshape_out", "layernorm_out"],
                                )
                                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "conv2d_input": [1, 128, 32, 32],
                "conv2d_filter": [128, 128, 1, 1],
                "elementwise_bias": [128],
                "layernorm_bias": [128],
                "layernorm_scale": [128],
            }
            self.dynamic_shape.max_input_shape = {
                "conv2d_input": [4, 320, 64, 64],
                "conv2d_filter": [320, 320, 1, 1],
                "elementwise_bias": [320],
                "layernorm_bias": [320],
                "layernorm_scale": [128],
            }
            self.dynamic_shape.opt_input_shape = {
                "conv2d_input": [4, 320, 64, 64],
                "conv2d_filter": [320, 320, 1, 1],
                "elementwise_bias": [320],
                "layernorm_bias": [320],
                "layernorm_scale": [128],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # just support dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (
            1e-2,
            1e-2,
        )  # tol 1e-2 for half

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()

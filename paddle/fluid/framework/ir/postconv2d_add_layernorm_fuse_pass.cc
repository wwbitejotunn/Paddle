// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/postconv2d_add_layernorm_fuse_pass.h"

#include <cmath>
#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                       \
  GET_IR_NODE(conv2d_00_op);              \
  GET_IR_NODE(conv2d_00_out);             \
  GET_IR_NODE(elementwise_add_10_op);     \
  GET_IR_NODE(elementwise_add_10_in_y);   \
  GET_IR_NODE(elementwise_add_10_out);    \
  GET_IR_NODE(reshapeLike_20_op);         \
  GET_IR_NODE(reshapeLike_20_out);        \
  GET_IR_NODE(transpose2_30_op);          \
  GET_IR_NODE(transpose2_30_out);         \
  GET_IR_NODE(transpose2_30_outXshape);         \
  GET_IR_NODE(layernorm_40_op);           \
  GET_IR_NODE(layernorm_40_bias);         \
  GET_IR_NODE(layernorm_40_scale);        \
  GET_IR_NODE(layernorm_40_out_y);

namespace paddle {
namespace framework {
namespace ir {

void PostConv2dAddLayernormFusePass::ApplyImpl(ir::Graph *graph) const {
    GraphPatternDetector gpd;
    FusePassBase::Init(scope_name_, graph);
    patterns::PostConv2dAddLayernormPattern pattern(
        gpd.mutable_pattern(),scope_name_);
    int fuse_count=0;
    // auto* scope = param_scope();
    PDNode* x = gpd.mutable_pattern()
            ->NewNode("x")
            ->assert_is_op_input("conv2d","Input")
            ->AsInput();
    pattern(x);
    auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g) {
        //TODO(wangboyun)
        // if (!IsCompat(subgraph, g)) {
            // return;
        // }
        VLOG(4)<<"post_conv2d_add_layernorm_fuse pass";
        GET_NODES;

        VarDesc post_conv2d_transpose2_out_desc(transpose2_30_out->Var()->Name());
        // post_conv2d_transpose2_out_desc->SetName("postConv2d_"+transpose2_30_out->Name());
        auto conv2d_out_shape = conv2d_00_out->Var()->GetShape();
        // post_conv2d_transpose2_out_desc->SetPersistable(true);
        post_conv2d_transpose2_out_desc.SetShape({conv2d_out_shape[0],
                                                  conv2d_out_shape[2],
                                                  conv2d_out_shape[3],
                                                  conv2d_out_shape[1]});
        post_conv2d_transpose2_out_desc.SetDataType(transpose2_30_out->Var()->GetDataType());
        post_conv2d_transpose2_out_desc.SetLoDLevel(transpose2_30_out->Var()->GetLoDLevel());
        auto* post_conv2d_transpose2_out_node = graph->CreateVarNode(&post_conv2d_transpose2_out_desc);
        auto transpose2_outXshape_shape = transpose2_30_outXshape->Var()->GetShape();
        transpose2_30_outXshape->Var()->SetShape({transpose2_outXshape_shape[0],
                                                  conv2d_out_shape[0],
                                                  conv2d_out_shape[2],
                                                  conv2d_out_shape[3],
                                                  conv2d_out_shape[1]});
        //todo

        VarDesc new_reshape_out_desc(reshapeLike_20_out->Var()->Name());
        new_reshape_out_desc.SetShape({conv2d_out_shape[0],
                                       conv2d_out_shape[2]*conv2d_out_shape[3],
                                       conv2d_out_shape[1]});
        new_reshape_out_desc.SetDataType(reshapeLike_20_out->Var()->GetDataType());
        new_reshape_out_desc.SetLoDLevel(reshapeLike_20_out->Var()->GetDataType());

        auto* new_reshape_out_node = graph->CreateVarNode(&new_reshape_out_desc);

        OpDesc post_conv2d_transpose2_op_desc(transpose2_30_op->Op()->Block());
        post_conv2d_transpose2_op_desc.SetType("transpose2");
        post_conv2d_transpose2_op_desc.SetInput("X",{conv2d_00_out->Name()});
        post_conv2d_transpose2_op_desc.SetOutput("Out",{post_conv2d_transpose2_out_node->Name()});
        post_conv2d_transpose2_op_desc.SetOutput("XShape",{transpose2_30_outXshape->Name()});
        post_conv2d_transpose2_op_desc.SetAttr("axis",std::vector<int>({0,2,3,1}));
        post_conv2d_transpose2_op_desc.Flush();

        auto* post_conv2d_transpose2_op_node = graph->CreateOpNode(&post_conv2d_transpose2_op_desc);

        OpDesc new_reshape_desc;
        new_reshape_desc.SetType("reshape2");
        new_reshape_desc.SetInput("X",{post_conv2d_transpose2_out_node->Name()});
        new_reshape_desc.SetOutput("Out",{new_reshape_out_node->Name()});  
        new_reshape_desc.SetAttr("shape",std::vector<int>{-1,
                                    static_cast<int>(conv2d_out_shape[2]*conv2d_out_shape[3]),
                                    static_cast<int>(conv2d_out_shape[1])});
        auto* new_reshape_node=graph->CreateOpNode(&new_reshape_desc);
        OpDesc prln_residual_bias_op_desc(elementwise_add_10_op->Op()->Block());

        prln_residual_bias_op_desc.SetType("preln_residual_bias");
        prln_residual_bias_op_desc.SetInput("X",{new_reshape_out_node->Name()});
        prln_residual_bias_op_desc.SetInput("EleBias",{elementwise_add_10_in_y->Name()});
        prln_residual_bias_op_desc.SetInput("Bias",{layernorm_40_bias->Name()});
        prln_residual_bias_op_desc.SetInput("Scale",{layernorm_40_scale->Name()});
        prln_residual_bias_op_desc.SetOutput("Out_0",{layernorm_40_out_y->Name()});
        prln_residual_bias_op_desc.SetAttr("Residual_num",0);
        prln_residual_bias_op_desc.SetAttr("epsilon", layernorm_40_op->Op()->GetAttr("epsilon"));
        prln_residual_bias_op_desc.SetAttr("begin_norm_axis",
                     layernorm_40_op->Op()->GetAttr("begin_norm_axis"));

        prln_residual_bias_op_desc.Flush();
        auto* post_conv2d_add_layernorm_op = graph->CreateOpNode(&prln_residual_bias_op_desc);

        IR_NODE_LINK_TO(conv2d_00_out,post_conv2d_transpose2_op_node);
        IR_NODE_LINK_TO(post_conv2d_transpose2_op_node,post_conv2d_transpose2_out_node);
        IR_NODE_LINK_TO(post_conv2d_transpose2_op_node,transpose2_30_outXshape);
        IR_NODE_LINK_TO(post_conv2d_transpose2_out_node,new_reshape_node);
        IR_NODE_LINK_TO(new_reshape_node,new_reshape_out_node);
        IR_NODE_LINK_TO(new_reshape_out_node,post_conv2d_add_layernorm_op);
        IR_NODE_LINK_TO(elementwise_add_10_in_y,post_conv2d_add_layernorm_op);
        IR_NODE_LINK_TO(layernorm_40_bias,post_conv2d_add_layernorm_op);
        IR_NODE_LINK_TO(layernorm_40_scale,post_conv2d_add_layernorm_op);
        IR_NODE_LINK_TO(post_conv2d_add_layernorm_op,layernorm_40_out_y);
        GraphSafeRemoveNodes(graph,
                            {elementwise_add_10_op,     
                            //  elementwise_add_10_in_y,   
                             elementwise_add_10_out,    
                             reshapeLike_20_op,        
                             reshapeLike_20_out,
                             transpose2_30_op,
                             transpose2_30_out,
                             layernorm_40_op
                            //  layernorm_40_bias,
                            //  layernorm_40_scale
                             });
        ++fuse_count;
    }; 
  gpd(graph, handler);
  AddStatis(fuse_count);

}

}
}
}

REGISTER_PASS(postconv2d_add_layernorm_fuse_pass,
              paddle::framework::ir::PostConv2dAddLayernormFusePass);
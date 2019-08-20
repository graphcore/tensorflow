# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu.vertex_edsl import PlaceholderVertexExpr
from tensorflow.python.ipu.vertex_edsl import DefaultNameSource


def codelet_expression_op(vertex_expression, *args):
  """
  Add a custom fused elementwise expression operation to the graph.

  Note that no autograd is done on this fused operation because the autograd
  code does not understand the internal structure of the fused codelet.

  :param vertex_expression: an vertex expression instance
  :param args: The arguments to the operation as a vector of Tensors
  :return: The Tensor which is a result of applying the elementwise operation
  """
  dtype = args[0].dtype
  placeholders = map(lambda x: PlaceholderVertexExpr("in" + str(x), None),
                     range(0, len(args)))
  concrete_expression = vertex_expression(*placeholders)
  expr = concrete_expression.lower(DefaultNameSource())
  return gen_poputil_ops.codelet_expression_op(
      input=args, dtype=dtype, source=expr)


def precompiled_user_op(inputs,
                        library_path,
                        gp_path=None,
                        outs=None,
                        name=None,
                        op_name=None):
  """
    Call the poplar function located in the shared library at 'library_path'
    as part of the normal tensorflow execution with the given 'inputs'. The
    shape and type of the output should be specified by 'outs' if it is None it
    will default to no output. 'outs' should be a dictionary with two elements
    like so:

    outs = {
          "output_types": [my_types_as_a_list],
          "output_shapes": [my_shapes_as_a_list],
      }
  """

  if outs is None:
    outs = {
        "output_types": [],
        "output_shapes": [],
    }
  gp_path = gp_path if gp_path else ""
  name = name if name else "UserOp"
  op_name = op_name if op_name else "Build"
  return gen_poputil_ops.ipu_user_op(
      inputs,
      library_path=library_path,
      gp_path=gp_path,
      op_name=op_name,
      name=name,
      is_gradient=False,
      **outs)

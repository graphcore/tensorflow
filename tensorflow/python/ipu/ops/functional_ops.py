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
"""
Functional operators
~~~~~~~~~~~~~~~~~~~~~~
"""
# Function captures are based on /tensorflow/python/ops/cond_v2.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util_v2 as util


class _InvalidCaptureException(Exception):
  pass


def _compile_function(func,
                      args,
                      scope,
                      control_outputs,
                      allow_external_captures=False):
  # Automatic control dependencies are added in defuns, but not in v1
  # graphs. Propagate that behavior here.
  add_control_dependencies = ops.get_default_graph()._add_control_dependencies  # pylint: disable=protected-access

  func_name = util.unique_fn_name(scope, "func")
  captured_args = [ops.convert_to_tensor(x) for x in args]

  # Compile the function to a graph.
  func_graph = func_graph_module.func_graph_from_py_func(
      func_name,
      func,
      captured_args, {},
      add_control_dependencies=add_control_dependencies)

  # Add the external captures (resources) to arguments.
  for t in func_graph.external_captures:
    if not allow_external_captures and t.dtype != dtypes.resource:
      raise _InvalidCaptureException(t.name)
  captured_args += func_graph.external_captures

  # Add any control outputs.  Autograph will add control outputs to the graph
  # automatically, so only add ones which are not already present.
  for o in control_outputs:
    if not o in func_graph.control_outputs:
      func_graph.control_outputs.extend([o])

  # Fix shape inference for the gradients and extract_outside_compilation_pass.
  for op in func_graph.get_operations():
    output_shapes = [out.get_shape() for out in op.outputs]
    # pylint: disable=protected-access
    op._set_shape_list_attr("_output_shapes", output_shapes)
    op._set_shape_list_attr("_xla_inferred_shapes", output_shapes)
    # pylint: enable=protected-access

  return func_graph, captured_args


def _convert_to_list(xs):
  if not isinstance(xs, (list, tuple)):
    return [xs]
  return list(xs)

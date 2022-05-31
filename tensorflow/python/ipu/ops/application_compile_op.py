# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import tempfile

from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops.functional_ops import _compile_function
from tensorflow.python.ipu.ops.functional_ops import _convert_to_list
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util_v2 as util


def experimental_application_compile_op(func,
                                        inputs=None,
                                        output_path=None,
                                        freeze_variables=False,
                                        name=None):
  """An operation that compiles a function into an executable for the IPU.
  The operation itself should be placed on CPU, and it will compile for the
  default IPU device.

  WARNING: This API is experimental and subject to change.

  Example usage:

  .. code-block:: python

    def model(x):
      return x * x

    v = tf.placeholder(tf.float32, shape=(2,))
    compile_model = experimental_application_compile_op(model, inputs=[v])

    with tf.Session() as sess:
      executable_path = sess.run(compile_model, {v: np.zeros(v.shape)})

  Args:
    func: The Python function to compile.
    inputs: The inputs passed to the function, as ``func(*inputs)``.
    output_path: The path where the executable will be stored. If None,
      a temporary file is used.
    freeze_variables: If True, any referenced variables will be captured
      by their values (when the compile op is executed) and embedded into
      the compiled executable as constants. If False, the referenced
      variables instead become implicit inputs that must be provided when
      executing the compiled executable.
    name: Optional op name.

  Returns:
    A `Tensor` of type string with the path to the compiled executable.
  """
  if inputs is None:
    inputs = []
  if name is None:
    name = "application_compile"
  if output_path is None:
    output_path = tempfile.mkstemp(prefix=name, suffix=".poplar_exec")[1]

  def wrapped_func(*args):
    ret = func(*args)
    if isinstance(ret, ops.Operation):
      # Returning operations is not supported.
      return None
    return ret

  inputs = _convert_to_list(inputs)

  with ops.name_scope(name) as scope:
    # Perform the `FuncGraph` expansion in XLA context.
    # This will for example ensure that v2 loops are used.
    xla_context = control_flow_ops.XLAControlFlowContext()
    try:
      xla_context.Enter()
      func_graph, captured_args = _compile_function(
          wrapped_func,
          inputs,
          scope, [],
          allow_external_captures=True,
          capture_by_value=freeze_variables)
    finally:
      xla_context.Exit()

  def get_op(op):
    while op.type == "Identity":
      op = op.inputs[0].op
    return op

  resource_indices = []
  constant_indices = []
  for index, arg in enumerate(captured_args):
    if arg.dtype == dtypes.resource:
      resource_indices.append(index)
    elif get_op(arg.op).type == "Const" or \
        (get_op(arg.op).type == "ReadVariableOp" and freeze_variables):
      constant_indices.append(index)

  with ops.control_dependencies(list(func_graph.control_captures)):
    return gen_application_runtime.ipu_application_compile(
        args=captured_args,
        resource_indices=resource_indices,
        constant_indices=constant_indices,
        function=util.create_new_tf_function(func_graph),
        executable_output_path=output_path,
        name=name)

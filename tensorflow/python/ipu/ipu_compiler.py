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
# =============================================================================
"""
Compiler interface
~~~~~~~~~~~~~~~~~~
"""

import collections
import inspect

from tensorflow_estimator.python.estimator import model_fn as estilib
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ipu import scopes as ipu_scope
from tensorflow.python.ipu import ipu_estimator


def compile(computation, inputs=None):
  """Builds an operator that compiles and runs `computation` with the Graphcore
  IPU XLA backend.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, `inputs` should be a list of n
      tensors.

      `computation` may return a list of operations and tensors.  Tensors must
      come before operations in the returned list.  The return value of
      `compile` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All operations returned from `computation` will be executed when
      evaluating any of the returned output tensors.
    inputs: A list of inputs or `None` (equivalent to an empty list). Each input
      can be a nested structure containing values that are convertible to
      tensors. Note that passing an N-dimension list of compatible values will
      result in a N-dimension list of scalar tensors rather than a single Rank-N
      tensors. If you need different behaviour, convert part of inputs to
      tensors with `tf.convert_to_tensor`.

  Returns:
    Same data structure as if `computation(inputs)` is called directly with some
    exceptions for correctness.

    1. None output. a NoOp would be returned which control-depends on
       computation.
    2. Single value output. A tuple containing the value would be returned.
    3. Operation-only outputs. a NoOp would be returned which
       control-depends on computation.

  Raises:
    Exception: If the computation was not compiled for an IPU device.
  """
  old_op_list = ops.get_default_graph().get_operations()
  try:
    with ipu_scope.ipu_jit_scope(0):
      result = xla.compile(computation, inputs)

    new_op_list = ops.get_default_graph().get_operations()

    added_ops = set(old_op_list) ^ set(new_op_list)
    # Go over all the new added ops, check that they have been placed on an IPU
    # device.
    placed_on_ipu = False
    all_no_ops = True
    for o in added_ops:
      device_spec = tf_device.DeviceSpec.from_string(o.device)
      if device_spec.device_type == 'IPU':
        placed_on_ipu = True
        break
      elif o.type != 'NoOp':
        all_no_ops = False

    if not placed_on_ipu and not all_no_ops:
      raise Exception("""\
  A computation has been compiled, however it was not placed on an IPU device. \
  This computation will not be executed on an IPU.
  To execute it on an IPU use the `ipu_scope` from `tensorflow.python.ipu.scopes`, \
  for example:

    with ipu_scope('/device:IPU:0'):
      result = ipu_compiler.compile(comp, inputs)
  """)
    return result

  except Exception as e:
    is_estimator = False
    try:
      # Retrieve the outputs of the computation from the trace
      outputs = inspect.trace()[-1][0].f_locals['outputs']
      is_estimator = _is_estimatorspec(outputs)
    except:
      raise e from None
    if is_estimator:
      raise ValueError("""\
  Your computation output contains an EstimatorSpec or IPUEstimatorSpec object.
  When you use an IPUEstimator, it already handles all the xla compilation
  and no manual call to compile() is needed.
  """)
    raise e


def _is_estimatorspec(outputs):
  """Checks if outputs contains an IPUEstimatorSpec or EstimatorSpec.

  Args:
    outputs: Object, tuple or list (can also be nested)

  Returns:
    True if an EstimatorSpec or IPUEstimatorSpec has been found
  """

  if isinstance(outputs,
                (ipu_estimator.IPUEstimatorSpec, estilib.EstimatorSpec)):
    return True
  # if the output is a list or a nested structure
  if isinstance(outputs, collections.Sequence):
    for o in outputs:
      if _is_estimatorspec(o):
        return True
  return False

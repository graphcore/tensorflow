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
Looping utilities
~~~~~~~~~~~~~~~~~
"""

# This implementation is based on:
# tensorflow/python/tpu/training_loop.py
# which creates the loops for the TPUs.

from tensorflow.python.framework import ops
from tensorflow.python.compiler.xla import xla
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow import TensorArray


def while_loop(condition,
               body,
               inputs=None,
               infeed_queue=None,
               maximum_iterations=None,
               use_while_v1=True):
  """Builds a while loop for IPUs.

  The set of loop-carried tensors corresponds to `inputs`.  Both
  `condition` and `body` take the current value of the loop-carried
  tensors. `condition` must return a single boolean value that determines
  whether iteration continues. `body` must return an updated list of values for
  the loop-carried tensors.

  Args:
    condition: a Python function that builds the loop condition.
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the loop, or
      None (equivalent to an empty list).
    infeed_queue: if not None, the IPUInfeedQueue from which data is consumed.
    use_while_v1: if True, then use a Tensorflow v1.x dataflow while loop.

  Returns:
    The final values of the loop-carried tensors.

  Raises:
    TypeError: if body or condition has the wrong signature.
  """

  # Converts inputs to Tensors if not TensorArray.
  inputs = [] if inputs is None else [
      ops.convert_to_tensor(x) if not isinstance(x, TensorArray) else x
      for x in _convert_to_list(inputs)
  ]
  input_types = [x.dtype for x in inputs]
  input_arity = len(inputs)
  body_arg_error = xla.check_function_argument_count(body, input_arity,
                                                     infeed_queue)
  if body_arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied loop body function cannot be called with the specified "
          "inputs. You specified %d inputs: %s, but the loop body needs %s." %
          (input_arity, str(inputs), body_arg_error))
    raise TypeError(
        "Supplied loop body function cannot be called with the specified "
        "inputs. You specified %d inputs: %s and %d additional inputs from "
        "infeed, but the computation needs %s." %
        (input_arity, str(inputs), infeed_queue.number_of_tuple_elements,
         body_arg_error))
  condition_arg_error = xla.check_function_argument_count(
      condition, input_arity, None)
  if condition_arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied loop condition function cannot be called with the "
          "specified inputs. You specified %d inputs: %s, but the loop "
          "condition needs %s." %
          (input_arity, str(inputs), condition_arg_error))
    raise TypeError(
        "Supplied loop condition function cannot be called with the "
        "specified inputs. You specified %d inputs: %s, but the loop "
        "condition needs %s. Note that infeed is not passed to the loop "
        "condition." % (input_arity, str(inputs), condition_arg_error))

  def condition_wrapper(*inputs):
    # Discards the dummy output added for arity-0 loops.
    if input_arity == 0:
      inputs = []
    return condition(*inputs)

  def body_wrapper(*inputs):
    """Wrapper around `body` that handles infeed queues and control deps."""
    inputs = list(inputs)

    # Discards the dummy output added for arity-0 loops.
    if input_arity == 0:
      inputs = []

    # Runs `body` with the dequeue_ops appended.
    if infeed_queue:
      dequeue_ops = infeed_queue._dequeue()
    else:
      dequeue_ops = []

    body_args, body_kwargs = _body_arguments(dequeue_ops)
    outputs = body(*(inputs + body_args), **body_kwargs)

    # If the computation only returned one value, make it a tuple.
    if not isinstance(outputs, (list, tuple)):
      outputs = (outputs,)

    outputs = [
        o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o)
        for o in outputs
    ]

    # Separates the returned Operations and Tensors.
    output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
    output_tensors = [o for o in outputs if not isinstance(o, ops.Operation)]

    if outputs != output_tensors + output_operations:
      raise ValueError("IPU loop body must return zero or more Tensor values "
                       "followed by zero or more Operations.")

    output_types = [op.dtype for op in output_tensors]
    if input_types != output_types:
      raise TypeError("Mismatch between input types and output types for loop "
                      "body: {} vs {}.".format(input_types, output_types))

    # Add a dummy output, if needed.
    if not output_tensors:
      output_tensors = [array_ops.constant(0)]

    if output_operations:
      output_tensors = control_flow_ops.tuple(output_tensors,
                                              control_inputs=output_operations)

    return output_tensors[0] if len(output_tensors) == 1 else output_tensors

  # If the body has arity 0, add a dummy loop-carried value to which we can add
  # control dependencies from any side-effecting operations.
  if input_arity == 0:
    inputs = [array_ops.constant(0)]

  if use_while_v1:
    while_fn = control_flow_ops.while_loop
  else:
    while_fn = while_v2.while_loop
    logging.warning("Usage of while_v2 is still experimental.")

  outputs = while_fn(condition_wrapper,
                     body_wrapper,
                     inputs,
                     maximum_iterations=maximum_iterations,
                     name="",
                     parallel_iterations=1)

  # Check the infeed queue has been used - this is more of a courtesy to the
  # user.
  if infeed_queue is not None and not infeed_queue.dequeued:
    raise ValueError("The infeed queue has not been dequeued.")

  outputs = _convert_to_list(outputs)
  if len(outputs) == 1:
    # If there were no inputs, only return the op for the dummy output.
    return outputs[0].op if input_arity == 0 else outputs[0]
  return outputs


def repeat(n, body, inputs=None, infeed_queue=None, use_while_v1=True):
  """Builds a loop that executes a fixed number of iterations.

  The set of loop-carried tensors correspond to `inputs`.
  `body` must be a function that takes and returns the values of the
  loop-carried tensors.

  Args:
    n: the number of loop iterations
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the loop or
      None (equivalent to an empty list).
    infeed_queue: if not None, the IPUInfeedQueue from which data is consumed.
    use_while_v1: if True, then use a Tensorflow v1.x dataflow while loop.
  Returns:
    The final values of the loop-carried tensors.
  Raises:
    ValueError: if there is a type error.
    TypeError: if body has the wrong signature.
  """
  if inputs is None:
    inputs = []

  input_arity = len(_convert_to_list(inputs))
  body_arg_error = xla.check_function_argument_count(body, input_arity,
                                                     infeed_queue)
  if body_arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied loop body function cannot be called with the specified "
          "inputs. You specified %d inputs: %s, but the loop body needs %s." %
          (input_arity, str(inputs), body_arg_error))
    raise TypeError(
        "Supplied loop body function cannot be called with the specified "
        "inputs. You specified %d inputs: %s and %d additional inputs from "
        "infeed, but the computation needs %s." %
        (input_arity, str(inputs), infeed_queue.number_of_tuple_elements,
         body_arg_error))
  return while_loop(lambda *args: True,
                    body,
                    inputs=inputs,
                    infeed_queue=infeed_queue,
                    maximum_iterations=n,
                    use_while_v1=use_while_v1)


def _convert_to_list(xs):
  if not isinstance(xs, (list, tuple)):
    return [xs]
  return list(xs)


def _body_arguments(inputs):
  """Returns the positional and named arguments for the loop body."""
  inputs = _convert_to_list(inputs)
  if len(inputs) == 1 and isinstance(inputs[0], dict):
    # A single dict is unpacked
    args = []
    kwargs = inputs[0]
  else:
    args = inputs
    kwargs = dict()
  return args, kwargs

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
"""
Embedded application runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import popef
from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
from tensorflow.compiler.plugin.poplar.ops import gen_dataset_exporters
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.driver import poplar_executable_pb2
from tensorflow.compiler.xla import xla_data_pb2

# A map from XLA protobuf types to tensorflow dtypes.
_type_map = {
    xla_data_pb2.PrimitiveType.S8: dtypes.int8,
    xla_data_pb2.PrimitiveType.S32: dtypes.int32,
    xla_data_pb2.PrimitiveType.F16: dtypes.float16,
    xla_data_pb2.PrimitiveType.F32: dtypes.float32,
    xla_data_pb2.PrimitiveType.U8: dtypes.uint8,
}


class RuntimeContext:
  """
  Represents an instance of the application runtime.

  This class must not be constructed directly, instead call
  `embedded_runtime_start` or `emedded_runtime_start_and_call`.
  """
  def __init__(self, name, executable_file, executable_proto, start_output):
    self._name = name
    self._executable_file = executable_file
    self._start_output = start_output
    self._executable_proto = executable_proto

  def start_output(self):
    """
    Get the output from the start op which will start the application runtime
    instance.

    Returns:
      The output tensor from the start op.
    """
    return self._start_output

  def name(self):
    """
    Get the name of the application runtime instance.

    Returns:
      The name of the application runtime instance.
    """
    return self._name

  def signature(self):
    """
    Get the signature of the executable.

    Returns:
      The signature protobuf object for the TF poplar executable.
    """
    return self._executable_proto.embedded_runtime_config.signature

  def output_types(self):
    """
    Get the output dtypes of the executable.

    Returns:
      A list of output dtypes for the TF poplar executable.
    """
    # Extract the outputs types and convert to TF dtypes, which are requered by the call API.
    return list(
        map(lambda output: _type_map[output.shape.element_type],
            self.signature().streamed_outputs))


def _find_opaque_blob(filename_tensor):
  opq_blobs = None

  # Get the serialized output.
  if executing_eagerly():
    path = filename_tensor.numpy()
  else:
    with ops.device("CPU"):
      with session_lib.Session().as_default():
        path = filename_tensor.eval()

  r = popef.Reader()
  r.parseFile(path)
  opq_blobs = r.opaqueBlobs()[0].data()
  return opq_blobs


def embedded_runtime_start(executable_file, inputs, name, timeout=None):
  """
  Create and start an application runtime from a TF poplar executable.

  Args:
    executable_file: The path to the executable file (given as string or Tensor)
    inputs: The initial input tensors.
    name: The name of the application runtime instance.
    timeout: An integer indicating how long (measured in microseconds)
      to allow an executable for a pipelined model or a model with IO tiles to
      wait for the next batch of data before forcing the execution to continue.
      This is required because pipelined models and models with IO tiles
      cannot proceed with execution until the next batch of data arrives. If not
      provided, defaults to 5000 microseconds.

  Returns:
    An embedded application runtime context instance.
  """

  # Check if the path to executable is constant (passed as string scalar)
  # or should be deduced in runtime from string Tensor passed in
  executable_file_tensor = ops.convert_to_tensor(executable_file,
                                                 dtype=dtypes.string)

  timeout = timeout or 5000

  # Open the executable file.
  opq_blob = _find_opaque_blob(executable_file_tensor)

  # Assert that the expected protobuf size is less than 128MB.
  # That is probably overkill, but it stops us trying to load HUGE files because we have nonsense data.
  assert len(opq_blob) < 128 * 1024 * 1024

  # Create the PoplarExecutableProto from the protobuf bytes.
  poplar_exec = poplar_executable_pb2.PoplarExecutableProto()
  poplar_exec.ParseFromString(opq_blob)

  if isinstance(inputs, dict):
    # Extract all the expected input names.
    names = list(
        map(lambda input: input.name,
            poplar_exec.embedded_runtime_config.signature.inputs))

    # Get the elements from the dictionary.
    inputs = list(map(lambda input_name: inputs.get(input_name, None), names))

    # Check we found a tensor for every input name.
    for input_name, input_tensor in zip(names, inputs):
      if input_tensor is None:
        raise Exception(
            f"Failed to find input tensor with name '{input_name}' in input "
            "dictionary.")

  # Convert tuples to lists.
  if isinstance(inputs, tuple):
    inputs = list(inputs)

  # Check we got a list of inputs.
  if not isinstance(inputs, list):
    raise Exception("Expected the inputs to be a list.")

  # Check we got enough inputs.
  if len(poplar_exec.embedded_runtime_config.signature.inputs) != len(inputs):
    raise Exception(
        f"Embedded application runtime expects "
        f"{len(poplar_exec.embedded_runtime_config.signature.inputs)} inputs, "
        f"but {len(inputs)} were provided.")

  # Check the shape and dtype of each input.
  for i, input_tensor in enumerate(inputs):
    actual_shape = list(input_tensor.shape)
    expected_shape = list(poplar_exec.embedded_runtime_config.signature.
                          inputs[i].shape.dimensions)
    name = poplar_exec.embedded_runtime_config.signature.inputs[i].name
    if actual_shape != expected_shape:
      raise Exception(
          f"Mismatched input shape at position {i} ('{name}'). Expected "
          f"{expected_shape}, but input {i} has shape {actual_shape}.")

    actual_dtype = inputs[i].dtype
    expected_dtype = _type_map[poplar_exec.embedded_runtime_config.signature.
                               inputs[i].shape.element_type]
    if expected_dtype != actual_dtype:
      raise Exception(
          f"Mismatched input dtype at position {i} ('{name}'). Expected "
          f"{expected_dtype}, but input {i} has dtype {actual_dtype}.")

  input_tensors = poplar_exec.embedded_runtime_config.signature.inputs
  arg2idx_map = {t.argument: i for i, t in enumerate(input_tensors)}
  reordered_inputs = [inputs[arg2idx_map[i]] for i in range(len(inputs))]

  # Create the context object that contains all the information required to
  # call the embedded runtime.
  app_runtime = gen_application_runtime.application_runtime(
      inputs=reordered_inputs,
      filename=executable_file_tensor,
      engine_name=name,
      timeout_us=timeout)
  return RuntimeContext(name, executable_file, poplar_exec, app_runtime)


def embedded_runtime_call(inputs, context):
  """
  Call an application with a batch of input data.

  Args:
    inputs: A batch of data to pass to the application.
    context: The application runtime context created with
             `embedded_runtime_start`.

  Returns:
    The output tensors from the application.
  """
  if len(inputs) != len(context.signature().streamed_inputs):
    raise Exception(f"Embedded application call expects "
                    f"{len(context.signature().streamed_inputs)} inputs, but "
                    f"{len(inputs)} were provided.")

  # Check the shape and dtype of each input.
  for i, input_tensor in enumerate(inputs):
    actual_shape = list(input_tensor.shape)
    expected_shape = list(
        context.signature().streamed_inputs[i].shape.dimensions)
    name = context.signature().streamed_inputs[i].name
    if actual_shape != expected_shape:
      raise Exception(
          f"Mismatched input shape at position {i} ('{name}'). Expected "
          f"{expected_shape}, but input {i} has shape {actual_shape}.")

    actual_dtype = inputs[i].dtype
    expected_dtype = _type_map[
        context.signature().streamed_inputs[i].shape.element_type]
    if expected_dtype != actual_dtype:
      raise TypeError(
          f"Mismatched input dtype at position {i} ('{name}'). Expected "
          f"{expected_dtype}, but input {i} has dtype {actual_dtype}.")

  # Use the context's start output, output types and engine name to call the
  # runtime.
  return gen_application_runtime.application_call(
      inputs,
      anchor=context.start_output(),
      outfeed_types=context.output_types(),
      engine_name=context.name())


def embedded_runtime_stop(context):  #pylint: disable=unused-argument
  """
  Stop an application runtime from a TF poplar executable.

  Args:
    context: The application runtime context created with
             `embedded_runtime_start`.
  """
  raise NotImplementedError(
      "Embedded runtime doesn't support stopping an engine.")


def embedded_runtime_start_and_call(executable_file, startup_inputs,
                                    call_inputs, name):
  """
  Create and start an application runtime from a TF poplar executable.

  Args:
    executable_file: The path to the executable file.
    startup_inputs: The initial input tensors.
    call_inputs: A batch of data to pass to the application.
    name: The name of the application runtime instance.

  Returns:
    A tuple of the batch results and the embedded application runtime context.
  """
  ctx = embedded_runtime_start(executable_file, startup_inputs, name)
  return (embedded_runtime_call(call_inputs, ctx), ctx)

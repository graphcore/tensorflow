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

import os
import tempfile
import numpy as np

from absl.testing import parameterized
from tensorflow.compiler.plugin.poplar.driver import poplar_executable_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu.config import DeviceConnectionType
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.ops.application_compile_op import experimental_application_compile_op as application_compile_op
from tensorflow.python.ipu.ops.embedded_runtime import _find_opaque_blob
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def parse_poplar_executable(executable_file):
  executable_file_tensor = ops.convert_to_tensor(executable_file,
                                                 dtype=dtypes.string)
  opq_blob = _find_opaque_blob(executable_file_tensor)

  poplar_exec = poplar_executable_pb2.PoplarExecutableProto()
  poplar_exec.ParseFromString(opq_blob)
  return poplar_exec


class TestApplicationCompile(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  def setUp(self):
    super().setUp()

    config = IPUConfig()
    config.auto_select_ipus = 1

    # Use the `NEVER` connection type to ensure that we never execute anything
    # with the Poplar executor, as we want compilation only. The `PRE_COMPILE`
    # connection type is slightly different in that it requires an executable
    # cache and allows a dummy excecution that populates its output with zeroes.
    config.device_connection.type = DeviceConnectionType.NEVER
    config.device_connection.version = "ipu2"

    # Disable the IPU model as it does not support executable serialization.
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS",
                                  "").replace("--use_ipu_model", "")
    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      config.configure_ipu_system()

  @test_util.deprecated_graph_mode_only
  def test_compile_simple_model(self):
    def my_net(x):
      return x * x

    v = array_ops.placeholder(dtype=np.float32, shape=(2,))
    result = application_compile_op(my_net, inputs=[v])
    self.assertEqual(result.dtype, dtypes.string)

    with session.Session() as sess:
      compiled_path = sess.run(result, {v: np.zeros(v.shape)})
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      self.assertEqual(len(signature.inputs), 1)
      self.assertEqual(len(signature.streamed_inputs), 0)
      self.assertEqual(len(signature.outputs), 1)
      self.assertEqual(len(signature.streamed_outputs), 0)

  @test_util.run_v2_only
  def test_compile_op_with_defunc_constant(self):
    @def_function.function
    def defunc():
      x = constant_op.constant([1.0, 1.0])
      return x * x

    compiled_path = application_compile_op(defunc,
                                           inputs=[],
                                           freeze_variables=True).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    self.assertEqual(len(signature.inputs), 0)  # constant embedded into graph
    self.assertEqual(len(signature.streamed_inputs), 0)
    self.assertEqual(len(signature.outputs), 1)  # return value
    self.assertEqual(len(signature.streamed_outputs), 0)

  @test_util.run_v2_only
  def test_compile_op_with_defunc_closure_constant(self):
    x = constant_op.constant([1.0, 1.0])

    @def_function.function
    def defunc():
      return x * x

    compiled_path = application_compile_op(defunc,
                                           inputs=[],
                                           freeze_variables=True).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    self.assertEqual(len(signature.inputs),
                     0)  # captured constant embedded into graph
    self.assertEqual(len(signature.streamed_inputs), 0)
    self.assertEqual(len(signature.outputs), 1)  # return value
    self.assertEqual(len(signature.streamed_outputs), 0)

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.run_v2_only
  def test_compile_op_with_defunc_module_resource(self, freeze_variables):
    class MyModule(module.Module):
      def __init__(self):
        super().__init__()
        self.v = variables.Variable([2.0, 2.0])

      @def_function.function
      def defunc(self):
        return self.v * self.v

    mod = MyModule()

    compiled_path = application_compile_op(
        mod.defunc, inputs=[], freeze_variables=freeze_variables).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    if freeze_variables:
      # Module variable embedded into executable as constant.
      self.assertEqual(len(signature.inputs), 0)
    else:
      # Module variable captured by reference as input to executable.
      self.assertEqual(len(signature.inputs), 1)
    self.assertEqual(len(signature.streamed_inputs), 0)
    self.assertEqual(len(signature.outputs), 1)  # return value
    self.assertEqual(len(signature.streamed_outputs), 0)

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.run_v2_only
  def test_compile_op_with_defunc_closure_resource(self, freeze_variables):

    v = variables.Variable([2.0, 2.0])

    @def_function.function(experimental_compile=True)
    def defunc():
      return v * v

    compiled_path = application_compile_op(
        defunc, freeze_variables=freeze_variables).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    if freeze_variables:
      # Closure variable embedded into executable as constant.
      self.assertEqual(len(signature.inputs), 0)
    else:
      # Closure variable captured by reference as input to executable.
      self.assertEqual(len(signature.inputs), 1)
    self.assertEqual(len(signature.streamed_inputs), 0)
    self.assertEqual(len(signature.outputs), 1)  # return value
    self.assertEqual(len(signature.streamed_outputs), 0)

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.run_v2_only
  def test_compile_op_with_defunc_closure_resource_cond(
      self, freeze_variables):
    ds = dataset_ops.Dataset.from_tensors((constant_op.constant(10.0,
                                                                shape=(3,),
                                                                name='x'),
                                           constant_op.constant(10.0,
                                                                shape=(),
                                                                name='p')))
    infeed = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    v = variables.Variable(3.0)

    @def_function.function(experimental_compile=True)
    def defunc():
      (x, p) = infeed._dequeue()  # pylint: disable=protected-access
      y = x * v
      if v < 0:
        z = y**2 + p
      elif v == 0:
        z = y * 2 - p
      else:
        z = y - 2
      res = z - x
      outfeed.enqueue(res)

    compiled_path = application_compile_op(
        defunc, freeze_variables=freeze_variables).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    if freeze_variables:
      # Closure variable embedded into executable as constant.
      self.assertEqual(len(signature.inputs), 0)
    else:
      # Closure variable captured by reference as input to executable.
      self.assertEqual(len(signature.inputs), 1)
    self.assertEqual(len(signature.streamed_inputs), 2)  # infeed._dequeue
    self.assertEqual(len(signature.outputs), 0)
    self.assertEqual(len(signature.streamed_outputs), 1)  # outfeed.enqueue

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.run_v2_only
  def test_compile_op_with_defunc_closure_resource_while(
      self, freeze_variables):
    ds = dataset_ops.Dataset.from_tensors(
        constant_op.constant(10.0, shape=(3,)))
    infeed = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    v = variables.Variable(3.0)

    @def_function.function(experimental_compile=True)
    def defunc(x):
      res = x * v * 10
      outfeed.enqueue(res)

    def predict_step_loop():
      r = loops.repeat(10, defunc, [], infeed)
      return r

    compiled_path = application_compile_op(
        predict_step_loop, freeze_variables=freeze_variables).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    if freeze_variables:
      # Closure variable embedded into executable as constant.
      self.assertEqual(len(signature.inputs), 0)
    else:
      # Closure variable captured by reference as input to executable.
      self.assertEqual(len(signature.inputs), 1)
    self.assertEqual(len(signature.streamed_inputs), 1)  # infeed._dequeue
    self.assertEqual(len(signature.outputs), 0)
    self.assertEqual(len(signature.streamed_outputs), 1)  # outfeed.enqueue

  @test_util.run_v2_only
  def test_compile_op_with_defunc_inputs(self):
    # The defunc's inputs must be supplied at compile-time.
    @def_function.function
    def defunc(x, y):
      return x * y

    with self.assertRaisesRegex(TypeError,
                                "missing 2 required positional arguments"):
      application_compile_op(defunc, inputs=[], freeze_variables=True)

  @test_util.run_v2_only
  def test_compile_op_with_defunc_infeed_outfeed(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        np.ones(10, dtype=np.float32))
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    @def_function.function
    def defunc():
      outfeed_queue.enqueue(infeed_queue._dequeue() * 2.0)  # pylint: disable=protected-access

    compiled_path = application_compile_op(defunc,
                                           inputs=[],
                                           freeze_variables=True).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    self.assertEqual(len(signature.inputs), 0)
    self.assertEqual(len(signature.streamed_inputs), 1)  # infeed._dequeue
    self.assertEqual(len(signature.outputs), 0)
    self.assertEqual(len(signature.streamed_outputs), 1)  # outfeed.enqueue

  @test_util.run_v2_only
  def test_compile_op_with_defunc_infeed_iterator_outfeed(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with ipu_strategy.IPUStrategy().scope():
      iterator = iter(
          dataset_ops.Dataset.from_tensor_slices(np.ones(10,
                                                         dtype=np.float32)))

    @def_function.function
    def defunc():
      for _ in math_ops.range(10):
        x = next(iterator)
        outfeed_queue.enqueue(x * 2.0)

    compiled_path = application_compile_op(defunc,
                                           inputs=[],
                                           freeze_variables=True).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    self.assertEqual(len(signature.inputs), 0)
    self.assertEqual(len(signature.streamed_inputs), 1)  # iterator
    self.assertEqual(len(signature.outputs), 0)
    self.assertEqual(len(signature.streamed_outputs), 1)  # outfeed.enqueue

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.run_v2_only
  def test_compile_op_with_defunc_pipeline_closure_resource(
      self, freeze_variables):
    dataset = dataset_ops.Dataset.from_tensor_slices((np.ones(
        (10, 5), dtype=np.float32),))
    dataset = dataset.batch(1, drop_remainder=True)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    v = variables.Variable([[2.0, 3.0, 4.0, 5.0, 6.0]], name='eagervar')

    def stage1(c, x):
      return c + x + v

    def stage2(x):
      return x + x

    @def_function.function
    def defunc():
      pipelining_ops.pipeline(computational_stages=[stage1, stage2],
                              gradient_accumulation_count=4,
                              infeed_queue=infeed_queue,
                              inputs=[42.0],
                              outfeed_queue=outfeed_queue,
                              device_mapping=[0, 0])

    compiled_path = application_compile_op(
        defunc, inputs=[], freeze_variables=freeze_variables).numpy()
    executable = parse_poplar_executable(compiled_path)
    signature = executable.embedded_runtime_config.signature

    if freeze_variables:
      self.assertEqual(len(signature.inputs),
                       0)  # closure variable baked in as constant
    else:
      self.assertEqual(len(signature.inputs),
                       1)  # closure variable becomes input

    self.assertEqual(len(signature.streamed_inputs), 1)  # pipeline infeed x
    self.assertEqual(len(signature.outputs), 0)
    self.assertEqual(len(signature.streamed_outputs), 1)  # pipeline outfeed

  @test_util.deprecated_graph_mode_only
  def test_compile_nonexistent_directory(self):
    output_path = os.path.join(self.get_temp_dir(), "nonexistent",
                               "file.poplar_exec")

    def my_net(x):
      return x * x

    v = array_ops.placeholder(dtype=np.float32, shape=(2,))
    result = application_compile_op(my_net,
                                    inputs=[v],
                                    output_path=output_path)

    with session.Session() as sess:
      with self.assertRaisesOpError("Failed to open file for writing"):
        sess.run(result, {v: np.zeros(v.shape)})

  @test_util.deprecated_graph_mode_only
  def test_compile_unsupported_dtype(self):
    with session.Session() as sess:

      def my_net(x):
        return x * x

      v = array_ops.placeholder(dtype=np.float64, shape=(2,))
      result = application_compile_op(my_net, inputs=[v])

      with self.assertRaisesOpError(
          "Detected unsupported operations when trying to compile graph"):
        sess.run(result, {v: np.zeros(v.shape)})

  @test_util.deprecated_graph_mode_only
  def test_compile_scalar_elementwise_graph(self):
    with session.Session() as sess:

      def my_net(x):
        return x + x

      v = array_ops.placeholder(dtype=np.float32, shape=())
      result = application_compile_op(my_net, inputs=[v])

      with self.assertRaisesOpError(
          "Cannot serialize a scalar elementwise graph"):
        sess.run(result, {v: np.zeros(v.shape)})

  @test_util.deprecated_graph_mode_only
  def test_compile_with_resource(self):
    with session.Session() as sess, tempfile.NamedTemporaryFile(
        dir=self.get_temp_dir()) as output_file:

      output_path = output_file.name

      def my_net(x):
        with variable_scope.variable_scope("vs", use_resource=True):
          w = variable_scope.get_variable("w", shape=(2,))
          return w * x

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      result = application_compile_op(my_net,
                                      inputs=[v],
                                      output_path=output_path)
      self.assertEqual(result.dtype, dtypes.string)

      sess.run(variables.global_variables_initializer())

      self.assertEqual(os.path.getsize(output_path), 0)
      compiled_path = sess.run(result, {v: np.zeros(v.shape)})
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      # We expect two inputs: The placeholder and the resource variable.
      self.assertEqual(len(signature.inputs), 2)
      self.assertEqual(len(signature.streamed_inputs), 0)
      self.assertEqual(len(signature.outputs), 1)
      self.assertEqual(len(signature.streamed_outputs), 0)

  @test_util.deprecated_graph_mode_only
  def test_compile_with_constant(self):
    with session.Session() as sess, tempfile.NamedTemporaryFile(
        dir=self.get_temp_dir()) as output_file:

      output_path = output_file.name

      def my_net(x, y):
        return x * y

      v = array_ops.placeholder(dtype=np.float32, shape=(2,))
      result = application_compile_op(my_net,
                                      inputs=[v, 42.0],
                                      output_path=output_path)
      self.assertEqual(result.dtype, dtypes.string)

      self.assertEqual(os.path.getsize(output_path), 0)
      compiled_path = sess.run(result, {v: np.zeros(v.shape)})
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      # We expect one input since the constant should be embedded.
      self.assertEqual(len(signature.inputs), 1)
      self.assertEqual(len(signature.streamed_inputs), 0)
      self.assertEqual(len(signature.outputs), 1)
      self.assertEqual(len(signature.streamed_outputs), 0)

  @test_util.deprecated_graph_mode_only
  def test_compile_infeed_and_outfeed(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        np.ones(10, dtype=np.float32))
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def body(v, x):
      v = v + x
      outfed = outfeed_queue.enqueue(v)
      return v, outfed

    def my_net(v):
      return loops.repeat(10, body, v, infeed_queue)

    result = application_compile_op(my_net, inputs=[0.0])

    with session.Session() as sess:
      compiled_path = sess.run(result)
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      # We expect zero inputs: The constant should be embedded.
      self.assertEqual(len(signature.inputs), 0)
      self.assertEqual(len(signature.streamed_inputs), 1)
      self.assertEqual(len(signature.outputs), 1)
      self.assertEqual(len(signature.streamed_outputs), 1)

  @test_util.deprecated_graph_mode_only
  def test_compile_training_loop(self):
    with session.Session() as sess:

      dataset = dataset_ops.Dataset.from_tensor_slices((np.ones(
          (10, 5), dtype=np.float32), np.ones((10, 1), dtype=np.float32)))
      dataset = dataset.batch(1, drop_remainder=True)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

      def body(lr, x, y):
        with variable_scope.variable_scope("vs", use_resource=True):
          predictions = layers.Dense(units=1)(x)

        loss = losses.mean_squared_error(labels=y, predictions=predictions)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss)
        return lr, train_op

      def my_net(lr):
        return loops.repeat(10, body, lr, infeed_queue)

      result = application_compile_op(my_net, inputs=[0.1])

      sess.run(variables.global_variables_initializer())
      compiled_path = sess.run(result)
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      # We expect two inputs: The two resource variables.
      self.assertEqual(len(signature.inputs), 2)
      self.assertEqual(len(signature.streamed_inputs), 2)
      self.assertEqual(len(signature.outputs), 3)
      self.assertEqual(len(signature.streamed_outputs), 0)

  @parameterized.named_parameters(("resources", False), ("constants", True))
  @test_util.deprecated_graph_mode_only
  def test_compile_pipeline(self, freeze_variables):
    with session.Session() as sess:

      dataset = dataset_ops.Dataset.from_tensor_slices((np.ones(
          (10, 5), dtype=np.float32),))
      dataset = dataset.batch(1, drop_remainder=True)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(offset, x):
        return layers.Dense(5, activation="relu")(x) + offset

      def stage2(x):
        return layers.Dense(10, activation="softmax")(x)

      def my_net():
        return pipelining_ops.pipeline(computational_stages=[stage1, stage2],
                                       gradient_accumulation_count=4,
                                       infeed_queue=infeed_queue,
                                       inputs=[42.0],
                                       outfeed_queue=outfeed_queue,
                                       device_mapping=[0, 0])

      result = application_compile_op(my_net,
                                      freeze_variables=freeze_variables)

      sess.run(variables.global_variables_initializer())
      compiled_path = sess.run(result)
      executable = parse_poplar_executable(compiled_path)
      signature = executable.embedded_runtime_config.signature

      if freeze_variables:
        self.assertEqual(len(signature.inputs), 0)
      else:
        self.assertEqual(len(signature.inputs),
                         len(variables.global_variables()))

      self.assertEqual(len(signature.streamed_inputs), 1)
      self.assertEqual(len(signature.outputs), 0)
      self.assertEqual(len(signature.streamed_outputs), 1)

  @test_util.deprecated_graph_mode_only
  def test_compile_op_placed_on_ipu(self):
    with session.Session() as sess:

      dataset = dataset_ops.Dataset.from_tensor_slices((np.ones(
          (10, 5), dtype=np.float32),))
      dataset = dataset.batch(1, drop_remainder=True)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(x):
        return layers.Dense(10, activation="relu")(x)

      def stage2(x):
        return layers.Dense(10, activation="softmax")(x)

      def my_net():
        return pipelining_ops.pipeline(computational_stages=[stage1, stage2],
                                       gradient_accumulation_count=4,
                                       infeed_queue=infeed_queue,
                                       outfeed_queue=outfeed_queue,
                                       device_mapping=[0, 0])

      with scopes.ipu_scope("/device:IPU:0"):
        result = application_compile_op(my_net, freeze_variables=True)

      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Cannot assign a device for operation application_compile"):
        sess.run(variables.global_variables_initializer())
        sess.run(result)


if __name__ == "__main__":
  test.main()

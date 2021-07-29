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

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.config import DeviceConnectionType
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ops.application_compile_op import experimental_application_compile_op as application_compile_op
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class TestApplicationCompile(test_util.TensorFlowTestCase):
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
      self.assertGreater(os.path.getsize(compiled_path), 0)

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
      self.assertEqual(compiled_path.decode(), output_path)
      self.assertGreater(os.path.getsize(output_path), 0)

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
      self.assertEqual(compiled_path.decode(), output_path)
      self.assertGreater(os.path.getsize(output_path), 0)

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
      self.assertGreater(os.path.getsize(compiled_path.decode()), 0)

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

      self.assertGreater(os.path.getsize(compiled_path.decode()), 0)


if __name__ == "__main__":
  test.main()

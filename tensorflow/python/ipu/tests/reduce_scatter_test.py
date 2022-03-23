# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.ops.all_to_all_op import all_gather
from tensorflow.python.ipu.ops.reduce_scatter_op import reduce_scatter
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ReduceScatterTest(test_util.TensorFlowTestCase):
  NUM_REPLICAS = 4

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReduceScatterSingleInput(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.NUM_REPLICAS
    cfg.configure_ipu_system()

    # Number of elements per-replica once the reduced data is scattered.
    output_size = 2
    # Number of elements in the input.
    input_size = self.NUM_REPLICAS * output_size

    # We are using the add collective op so we generate data with unique powers
    # of 2 to make sure all combinations are unique.
    data = np.asarray(range(self.NUM_REPLICAS * input_size), dtype=np.float32)
    data = 2**data

    # One batch of shape [input_size] for each replica.
    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size=input_size, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      x = reduce_scatter(x, self.NUM_REPLICAS, 'COLLECTIVE_OP_ADD')
      return all_gather(x, self.NUM_REPLICAS)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res, = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)

    # Calculate expected output with numpy.
    # Split data by replica.
    expected = data.reshape([self.NUM_REPLICAS, input_size])
    # Reduce across replica dimension.
    expected = np.sum(expected, axis=0)
    # Split results between replicas (scatter).
    expected = expected.reshape([self.NUM_REPLICAS, output_size])

    self.assertAllEqual(expected, result)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReduceScatterMultipleInputs(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.NUM_REPLICAS
    cfg.configure_ipu_system()

    # Number of elements per-replica once the reduced data is scattered.
    output_size = 2
    # Number of elements in each input.
    input_size = self.NUM_REPLICAS * output_size
    # Number of inputs.
    num_inputs = 2

    # We are using the add collective op so we generate data with unique powers
    # of 2 to make sure all combinations are unique.
    data = np.asarray(range(self.NUM_REPLICAS * input_size * num_inputs),
                      dtype=np.float32)
    data = 2**data

    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    # A batch of shape [num_inputs, input_size] for each replica.
    dataset = dataset.batch(batch_size=input_size, drop_remainder=True)
    dataset = dataset.batch(batch_size=num_inputs, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      # Split infeed input and reduce_scatter with each as a separare input.
      x1, x2 = array_ops.unstack(x, axis=0)
      x1, x2 = reduce_scatter([x1, x2], self.NUM_REPLICAS, 'COLLECTIVE_OP_ADD')
      return all_gather([x1, x2], self.NUM_REPLICAS)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)

    # Calculate expected output with numpy.
    # Arrange data by replica and input.
    expected = data.reshape([self.NUM_REPLICAS, num_inputs, input_size])
    # Reduce across replica dimension.
    expected = np.sum(expected, axis=0)
    # Split results between replicas (scatter).
    expected = expected.reshape([num_inputs, self.NUM_REPLICAS, output_size])

    self.assertAllEqual(expected, result)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReduceScatterMixedDtypes(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.NUM_REPLICAS
    cfg.configure_ipu_system()

    # Number of elements per-replica once the reduced data is scattered.
    output_size = 8
    # Number of elements in each input.
    input_size = self.NUM_REPLICAS * output_size
    # Number of inputs.
    num_inputs = 2

    # Use seeded random values for the data instead of powers of 2 because of
    # the limited precision of the dtypes.
    rng = np.random.default_rng(seed=123)
    data = rng.integers(low=0,
                        high=100,
                        dtype=np.int32,
                        size=[self.NUM_REPLICAS * input_size * num_inputs])

    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    # A batch of shape [num_inputs, input_size] for each replica.
    dataset = dataset.batch(batch_size=input_size, drop_remainder=True)
    dataset = dataset.batch(batch_size=num_inputs, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      # Split infeed input and reduce_scatter with each as a separare input.
      x1, x2 = array_ops.unstack(x, axis=0)
      # Perform cast on one input so each input has a different dtype.
      x2 = math_ops.cast(x2, np.float16)
      x1, x2 = reduce_scatter([x1, x2], self.NUM_REPLICAS, 'COLLECTIVE_OP_ADD')
      return all_gather([x1, x2], self.NUM_REPLICAS)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)

    # Calculate expected output with numpy.
    # Arrange data by replica and input.
    expected = data.reshape([self.NUM_REPLICAS, num_inputs, input_size])
    # Reduce across replica dimension.
    expected = np.sum(expected, axis=0)
    # Split results between replicas (scatter).
    expected = expected.reshape([num_inputs, self.NUM_REPLICAS, output_size])

    self.assertAllEqual(expected, result)
    # Input dtypes are preserved in the outputs.
    self.assertEqual(result[0].dtype, np.int32)
    self.assertEqual(result[1].dtype, np.float16)


if __name__ == "__main__":
  googletest.main()

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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.ops.all_to_all_op import all_gather


class AllGatherTest(test_util.TensorFlowTestCase):
  REPLICATION_FACTOR = 4

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testAllGatherSingleInput(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.REPLICATION_FACTOR
    cfg.configure_ipu_system()

    data = np.asarray(range(self.REPLICATION_FACTOR * 8), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    # One batch of shape [8] for each replica:
    # [[0, 1,  2,  3,  4,  5,  6,  7],
    #  [8, 9, 10, 11, 12, 13, 14, 15], ...
    dataset = dataset.batch(batch_size=8, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      return all_gather(x, self.REPLICATION_FACTOR)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res, = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
    expected = data.reshape([self.REPLICATION_FACTOR, 8])
    # Expected output the same as input, as the batches from each replica are
    # reassembled by the all_gather.
    self.assertAllEqual(result, expected)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testAllGatherMultipleInputs(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.REPLICATION_FACTOR
    cfg.configure_ipu_system()

    data = np.asarray(range(self.REPLICATION_FACTOR * 32), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    # One batch of shape [4, 8] for each replica.
    dataset = dataset.batch(batch_size=8, drop_remainder=True)
    dataset = dataset.batch(batch_size=4, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      # Split infeed input into 4 and all_gather with each as a separare input.
      x1, x2, x3, x4 = array_ops.unstack(x, axis=0)
      return all_gather([x1, x2, x3, x4], self.REPLICATION_FACTOR)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
    # Expected output to be a transposed version of the input.
    # Each of the 4 all_gather outputs should be of shape
    # [REPLICATION_FACTOR, 8].
    expected = data.reshape([self.REPLICATION_FACTOR, 4, 8])
    expected = np.transpose(expected, [1, 0, 2])
    self.assertAllEqual(result, expected)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testAllGatherMixedDtypes(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = self.REPLICATION_FACTOR
    cfg.configure_ipu_system()

    data = np.asarray(range(self.REPLICATION_FACTOR * 16), dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    # One batch of shape [2, 8] for each replica.
    dataset = dataset.batch(batch_size=8, drop_remainder=True)
    dataset = dataset.batch(batch_size=2, drop_remainder=True)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      x = infeed_queue._dequeue()  # pylint: disable=protected-access
      # Split infeed input into 2 and all_gather with each as a separare input.
      x1, x2 = array_ops.unstack(x, axis=0)
      x2 = math_ops.cast(x2, np.int32)
      return all_gather([x1, x2], self.REPLICATION_FACTOR)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res = ipu.ipu_compiler.compile(my_net)

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
    # Expected output to be a transposed version of the input.
    # Both of the all_gather outputs should be of shape [REPLICATION_FACTOR, 8].
    expected = data.reshape([self.REPLICATION_FACTOR, 2, 8])
    expected = np.transpose(expected, [1, 0, 2])
    self.assertAllEqual(result, expected)
    # Input dtypes are preserved in the outputs.
    self.assertEqual(result[0].dtype, np.float16)
    self.assertEqual(result[1].dtype, np.int32)


if __name__ == "__main__":
  googletest.main()

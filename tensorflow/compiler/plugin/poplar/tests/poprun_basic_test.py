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
import numpy as np
from tensorflow.python.ipu.config import IPUConfig

import popdist

from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.ops.cross_replica_ops import cross_replica_sum
from tensorflow.python.ipu.ops.replication_ops import replication_index
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PoprunBasicTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @test_util.deprecated_graph_mode_only
  def test_cross_replica_sum(self):
    num_local_replicas = popdist.getNumLocalReplicas()
    num_total_replicas = popdist.getNumTotalReplicas()
    data = np.array([0.0] * num_local_replicas, dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((data,))
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_body(acc, x):
      index = math_ops.cast(replication_index(), np.float32)
      acc += cross_replica_sum(index)
      acc += x * x
      return acc, outfeed.enqueue(index)

    def my_net():
      return ipu.loops.repeat(1, my_body, infeed_queue=infeed, inputs=[0.0])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      [result] = ipu.ipu_compiler.compile(my_net)

    init_op = infeed.initializer
    dequeue_op = outfeed.dequeue()

    with session.Session() as sess:
      config = IPUConfig()
      config.select_ipus = [popdist.getDeviceId()]

      config.experimental.multi_replica_distribution.process_count = \
          popdist.getNumInstances()
      config.experimental.multi_replica_distribution.process_index = \
          popdist.getInstanceIndex()

      config.configure_ipu_system()

      sess.run(init_op)
      res = sess.run(result)
      [dequeued] = sess.run(dequeue_op)

      replica_offset = popdist.getReplicaIndexOffset()
      self.assertAllEqual(
          dequeued,
          np.arange(replica_offset, replica_offset + num_local_replicas))
      self.assertEqual(res, np.sum(np.arange(num_total_replicas)))


if __name__ == "__main__":
  test.main()

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.ops.cross_replica_ops import cross_replica_sum
from tensorflow.python.ipu.ops.replication_ops import replication_index
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# Prerequisite for running this test on a POD system:
#
# Create a partition with two IPUs and two GCDs:
# $ vipu-admin create partition part --size 2 --num-gcds 2 --cluster clust
#
# After that you should have two IPUs like this:
# $ gc-info --list-devices
# Graphcore device listing:
#
# -+- Id: [0], target: [Fabric], PCI Domain: [3]
# -+- Id: [1], target: [Fabric], PCI Domain: [2]


class MultiReplicaDistributionTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @test_util.deprecated_graph_mode_only
  def test_cross_replica_sum(self):
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def my_net(x):
      y = math_ops.cast(replication_index(), np.float32) + 1.0
      y = cross_replica_sum(y)
      return x * x + y

    v = array_ops.placeholder(dtype=np.float32, shape=(1,))
    with ipu.scopes.ipu_scope("/device:IPU:0"):
      [result] = ipu.ipu_compiler.compile(my_net, inputs=[v])

    with session.Session() as sess:
      config = ipu.utils.create_ipu_config()
      config = ipu.utils.auto_select_ipus(config, 1)

      config = ipu.utils.set_experimental_multi_replica_distribution_options(
          config, process_count=size, process_index=rank)

      ipu.utils.configure_ipu_system(config)

      res = sess.run(result, {v: [2.0]})
      self.assertEqual(res, 4.0 + np.sum(range(size + 1)))


if __name__ == "__main__":
  test.main()

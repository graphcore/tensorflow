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
# =============================================================================
import numpy as np
from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops


class TestReplicatedReduceScatter(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):
  @parameterized.named_parameters(
      ('reduce_add', 'COLLECTIVE_OP_ADD', 4),
      ('reduce_mean', 'COLLECTIVE_OP_MEAN', 1),
  )
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_reduce_scatter(self, op, scale):
    with session_lib.Session() as sess:
      num_replicas = 4

      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(*xs):
        y = [
            ipu.ops.reduce_scatter_op.reduce_scatter(
                x, replication_factor=num_replicas, op=op) for x in xs
        ]
        return outfeed_queue.enqueue(y)

      inputs = [i * np.arange(i, dtype=np.float32) for i in range(1, 6)]
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_net = ipu.ipu_compiler.compile(my_net, inputs=inputs)

      gathered = []
      with ops.device("/device:CPU:0"):
        dequeued = outfeed_queue.dequeue()
        for scattered in dequeued:
          gathered.append(array_ops.reshape(scattered, shape=[-1]))

      cfg = ipu.config.IPUConfig()
      cfg.optimizations.maximum_reduce_scatter_buffer_size = 10000
      cfg.auto_select_ipus = num_replicas
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      sess.run(compiled_net)
      out = sess.run(gathered)

      # Check padded lengths.
      self.assertEqual(len(out[0]), np.ceil(1 / num_replicas) * num_replicas)
      self.assertEqual(len(out[1]), np.ceil(2 / num_replicas) * num_replicas)
      self.assertEqual(len(out[2]), np.ceil(3 / num_replicas) * num_replicas)
      self.assertEqual(len(out[3]), np.ceil(4 / num_replicas) * num_replicas)
      self.assertEqual(len(out[4]), np.ceil(5 / num_replicas) * num_replicas)

      # Check payloads.
      self.assertAllEqual(1.0 * scale * np.arange(1), out[0][:1])
      self.assertAllEqual(2.0 * scale * np.arange(2), out[1][:2])
      self.assertAllEqual(3.0 * scale * np.arange(3), out[2][:3])
      self.assertAllEqual(4.0 * scale * np.arange(4), out[3][:4])
      self.assertAllEqual(5.0 * scale * np.arange(5), out[4][:5])


if __name__ == "__main__":
  googletest.main()

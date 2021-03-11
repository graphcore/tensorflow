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
# =============================================================================

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.scopes import ipu_shard
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GclTest(test_util.TensorFlowTestCase):
  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testIoTilesAreExcludedFromShard(self):
    def my_net(a, b):
      with ipu_shard(0):
        aa = math_ops.matmul(a, a, transpose_b=True, name="aa")
      with ipu_shard(1):
        bb = math_ops.matmul(b, b, transpose_b=True, name="bb")
      return aa, bb

    input_a = array_ops.placeholder(np.float32, [1216, 1])
    input_b = array_ops.placeholder(np.float32, [1216, 1])

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[input_a, input_b])

    num_io_tiles = 128
    cfg = ipu_utils.create_ipu_config(profiling=True)
    cfg = ipu_utils.set_io_tile_options(cfg, num_io_tiles=num_io_tiles)
    cfg = ipu_utils.auto_select_ipus(cfg, num_ipus=2)
    ipu_utils.configure_ipu_system(cfg)

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(compiled_net, {
          input_a: np.ones(input_a.shape),
          input_b: np.ones(input_b.shape)
      })

      report.parse_log()
      num_compute_tiles = report.get_num_tiles_per_ipu() - num_io_tiles
      for t in report.get_tensor_map().all_tensors():
        self.assertLessEqual(len(t.tiles), num_compute_tiles)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReplicated(self):
    num_replicas = 2
    num_shards = 2

    def my_model(x):
      with ipu.scopes.ipu_shard(0):
        y = x + math_ops.cast(ipu.ops.replication_ops.replication_index(),
                              np.float32)
      with ipu.scopes.ipu_shard(1):
        z = ipu.ops.cross_replica_ops.cross_replica_sum(y,
                                                        name="my_all_reduce")
      return z

    inputs = array_ops.placeholder(np.float32, shape=[])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      compiled_model = ipu.ipu_compiler.compile(my_model, [inputs])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.auto_select_ipus(cfg, num_replicas * num_shards)
    cfg = ipu.utils.set_gcl_options(cfg,
                                    gcl_options={
                                        "useSynclessCollectives": "true",
                                    })
    cfg = ipu.utils.set_io_tile_options(cfg, num_io_tiles=128)
    cfg = tu.add_hw_ci_connection_options(cfg)
    ipu.utils.configure_ipu_system(cfg)

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()
      [result] = sess.run(compiled_model, {inputs: 1.0})
      self.assertEqual(result, sum(range(1, num_replicas + 1)))
      report.parse_log()
      report.assert_compute_sets_contain_list(
          ["my_all_reduce/all-reduce.*/AllReduceGCL"])


if __name__ == "__main__":
  test.main()

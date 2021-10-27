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
import pva

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
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
    tiles_per_ipu = 1216
    num_io_tiles = 128
    num_compute_tiles = tiles_per_ipu - num_io_tiles

    a = np.ones([tiles_per_ipu, 1], dtype=np.float32)
    b = np.ones([tiles_per_ipu, 1], dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors((a, b))
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net():
      with ipu_shard(0):
        # I/O op to ensure I/O tiles are allocated for this shard
        a, b = infeed._dequeue()  # pylint: disable=protected-access
        aa = math_ops.matmul(a, a, transpose_b=True, name="aa")
      with ipu_shard(1):
        bb = math_ops.matmul(b, b, transpose_b=True, name="bb")
        # I/O op to ensure I/O tiles are allocated for this shard
        return outfeed.enqueue((aa, bb))

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu
    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.auto_select_ipus = 2
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with session.Session() as sess:
      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      sess.run(infeed.initializer)
      sess.run(compiled_net)
      sess.run(outfeed.dequeue())

      report_json.parse_log()
      for t in report_json.get_tensor_map().all_tensors():
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

    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.auto_select_ipus = num_replicas * num_shards
    cfg.gcl_poplar_options = {'useSynclessCollectives': 'true'}
    cfg.io_tiles.num_io_tiles = 128
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with session.Session() as sess:
      [result] = sess.run(compiled_model, {inputs: 1.0})

    self.assertEqual(result, sum(range(1, num_replicas + 1)))
    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_contain_list(
        report, ["my_all_reduce/all-reduce.*/AllReduceGCL"])


if __name__ == "__main__":
  test.main()

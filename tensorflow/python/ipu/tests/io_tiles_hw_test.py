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

import itertools
import numpy as np
from absl.testing import parameterized
import pva
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class IoTilesHWTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters(
      itertools.product([8, 16, 32, 64, 128], [True, False]))
  @test_util.deprecated_graph_mode_only
  @tu.test_uses_ipus(num_ipus=1)
  def testSingleIPUIOOverlap(self, num_io_tiles, use_io_tiles):
    data = np.ones((num_io_tiles, 512), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors(data)
    dataset = dataset.repeat(16)
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    def body(a):
      with ops.name_scope("matmul"):
        ab = math_ops.matmul(a, a, transpose_b=True)
      with ops.name_scope("outfeed"):
        return outfeed.enqueue(ab)

    def loop():
      return loops.repeat(16, body, [], infeed)

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(loop, inputs=[])

    cfg = IPUConfig()

    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access

    if use_io_tiles:
      cfg.io_tiles.num_io_tiles = num_io_tiles
      cfg.io_tiles.place_ops_on_io_tiles = True

    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with session.Session() as sess:
      sess.run(infeed.initializer)
      report_helper.clear_reports()
      sess.run(compiled_net)

    report = pva.openReport(report_helper.find_report())

    if use_io_tiles:
      self.assert_compute_io_overlap_percentage(report, 0.9)
    else:
      with self.assertRaisesRegex(AssertionError,
                                  r"0\.0 not greater than 0\.9"):
        self.assert_compute_io_overlap_percentage(report, 0.9)


if __name__ == "__main__":
  test.main()

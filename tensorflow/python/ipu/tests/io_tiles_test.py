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

from absl.testing import parameterized
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class IoTilesTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters([True, False])
  @test_util.deprecated_graph_mode_only
  def testTensorPlacementAndExchanges(self, buffer_fits_on_io_tiles):
    tiles_per_ipu = 1216
    num_io_tiles = 32
    num_compute_tiles = tiles_per_ipu - num_io_tiles
    proportion = 100 if buffer_fits_on_io_tiles else 0.1

    data = np.ones((tiles_per_ipu, tiles_per_ipu), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors((data, data))
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="infeed")
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

    def my_net():
      with ops.name_scope("infeed"):
        a, b = infeed._dequeue()  # pylint: disable=protected-access
      with ops.name_scope("matmul"):
        ab = math_ops.matmul(a, b)
      with ops.name_scope("outfeed"):
        return outfeed.enqueue(ab)

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu_utils.create_ipu_config(profiling=True)
    cfg = ipu_utils.set_ipu_model_options(cfg,
                                          compile_ipu_code=False,
                                          tiles_per_ipu=tiles_per_ipu)

    cfg = ipu_utils.set_io_tile_options(
        cfg,
        num_io_tiles=num_io_tiles,
        place_ops_on_io_tiles=True,
        io_tile_available_memory_proportion=proportion)

    cfg = ipu_utils.auto_select_ipus(cfg, num_ipus=1)
    ipu_utils.configure_ipu_system(cfg)

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(infeed.initializer)
      sess.run(compiled_net)

      [result] = sess.run(outfeed.dequeue())
      self.assertAllEqual(np.dot(data, data), result)

      report.parse_log()

      # Check that the tensors are mapped as expected.
      io_tensors = []
      compute_tensors = []
      for t in report.get_tensor_map().all_tensors():
        if t.name.startswith("infeed") or t.name.startswith("outfeed"):
          io_tensors.append(t)
        elif t.name.startswith("matmul"):
          compute_tensors.append(t)

      self.assertGreater(len(io_tensors), 0)

      for t in io_tensors:
        if buffer_fits_on_io_tiles:
          self.assertLessEqual(len(t.tiles), num_io_tiles)
        else:
          self.assertLessEqual(len(t.tiles), num_compute_tiles)

      self.assertGreater(len(compute_tensors), 0)
      for t in compute_tensors:
        self.assertLessEqual(len(t.tiles), num_compute_tiles)

      # Check that the expected inter-tileset exchanges are performed.
      exchanges = report.get_program_names_of_type("DoExchange")
      expected_exchanges = [
          "infeed/*/inter-tileset-copy",  # copy from IO tiles to compute tiles
          "matmul/*/inter-tileset-copy",  # copy from compute tiles to IO tiles
      ]
      if buffer_fits_on_io_tiles:
        self.assertFalse(
            tu.missing_whitelist_entries_in_names(exchanges,
                                                  expected_exchanges),
            exchanges)

  @test_util.deprecated_graph_mode_only
  def testTensorLayoutOnIoTiles(self):
    tiles_per_ipu = 1216
    num_io_tiles = 32

    data = np.arange(128, dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensors(data)
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="infeed2")

    def my_net():
      with ops.name_scope("infeed"):
        a = infeed._dequeue()  # pylint: disable=protected-access
      with ops.name_scope("multiply"):
        aa = math_ops.multiply(a, a)
      return aa

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu_utils.create_ipu_config(profiling=True)
    cfg = ipu_utils.set_ipu_model_options(cfg,
                                          compile_ipu_code=False,
                                          tiles_per_ipu=tiles_per_ipu)

    cfg = ipu_utils.set_io_tile_options(cfg,
                                        num_io_tiles=num_io_tiles,
                                        place_ops_on_io_tiles=True)

    cfg = ipu_utils.auto_select_ipus(cfg, num_ipus=1)
    ipu_utils.configure_ipu_system(cfg)

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(infeed.initializer)
      [result] = sess.run(compiled_net)
      self.assertAllEqual(data * data, result)

      report.parse_log()

      # Find the IO tile tensors.
      io_tensors = []
      for t in report.get_tensor_map().all_tensors():
        if t.name.startswith("infeed"):
          io_tensors.append(t)
      self.assertGreater(len(io_tensors), 0)

      # The IO tensors are so small that they should be mapped as a host packet to a single tile.
      for t in io_tensors:
        self.assertEqual(len(t.tiles), 1)
        self.assertEqual(t.tiles[0].num_elements, len(data))


if __name__ == "__main__":
  test.main()

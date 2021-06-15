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
from tensorflow.python.ipu.scopes import ipu_shard
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class IoTilesTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters([True, False])
  @test_util.deprecated_graph_mode_only
  @tu.test_may_use_ipus_or_model(num_ipus=1)
  def testTensorPlacementAndExchanges(self, buffer_fits_on_io_tiles):
    tiles_per_ipu = 1216
    num_io_tiles = 32
    num_compute_tiles = tiles_per_ipu - num_io_tiles
    proportion = 100.0 if buffer_fits_on_io_tiles else 0.1

    data = np.ones((tiles_per_ipu, tiles_per_ipu), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors((data, data))
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net():
      with ops.name_scope("infeed"):
        a, b = infeed._dequeue()  # pylint: disable=protected-access
      with ops.name_scope("matmul"):
        ab = math_ops.matmul(a, b)
      with ops.name_scope("outfeed"):
        return outfeed.enqueue(ab)

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu

    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.io_tiles.available_memory_proportion = proportion

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

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
  @tu.test_may_use_ipus_or_model(num_ipus=1)
  def testTensorLayoutOnIoTiles(self):
    tiles_per_ipu = 1216
    num_io_tiles = 32

    data = np.arange(128, dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensors(data)
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      with ops.name_scope("infeed"):
        a = infeed._dequeue()  # pylint: disable=protected-access
      with ops.name_scope("multiply"):
        aa = math_ops.multiply(a, a)
      return aa

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu

    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

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

  @test_util.deprecated_graph_mode_only
  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def testIoTilesOnlyCreatedOnDevicesWithIoInstructions(self):
    tiles_per_ipu = 1216
    num_io_tiles = 32
    num_compute_tiles = tiles_per_ipu - num_io_tiles

    data = np.ones((tiles_per_ipu, tiles_per_ipu), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors(data)
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      with ipu_shard(0):
        with ops.name_scope("infeed"):
          a = infeed._dequeue()  # pylint: disable=protected-access
        # Matmul on the same device as an infeed should be limited to
        # num_compute_tiles tiles as IO tiles are allocated on this device.
        with ops.name_scope("matmul1"):
          aa = math_ops.matmul(a, a)
      with ipu_shard(1):
        with ops.name_scope("matmul2"):
          # Matmul on a device with no IO should limited to tiles_per_ipu tiles
          # as no IO tiles should be allocated on this device.
          aaa = math_ops.matmul(aa, a)
      return aaa

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu
    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(infeed.initializer)
      result, = sess.run(compiled_net)
      report.parse_log()

      # Sanity check results.
      self.assertAllEqual(np.dot(np.dot(data, data), data), result)

      # Find matmul operations.
      tensors = report.get_tensor_map().all_tensors()
      matmuls = [t for t in tensors if t.id.startswith("dot")]
      self.assertEqual(len(matmuls), 2)

      # Matmul on ipu0 mapped to a smaller set of tiles.
      self.assertLessEqual(len(matmuls[0].tiles), num_compute_tiles)

      # Matmul on ipu1 device allowed to use all tiles.
      self.assertGreater(len(matmuls[1].tiles), num_compute_tiles)
      self.assertLessEqual(len(matmuls[1].tiles), tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def testCopyFromDeviceWithIoTilesToDeviceWithoutIoTiles(self):
    tiles_per_ipu = 1216
    num_io_tiles = 32
    num_compute_tiles = tiles_per_ipu - num_io_tiles

    data = np.ones((tiles_per_ipu, tiles_per_ipu), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensors(data)
    infeed = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      with ipu_shard(0):
        with ops.name_scope("infeed"):
          a = infeed._dequeue()  # pylint: disable=protected-access
        with ops.name_scope("matmul1"):
          a = math_ops.matmul(a, a)
      with ipu_shard(1):
        with ops.name_scope("matmul2"):
          return math_ops.matmul(a, a)

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[])

    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu
    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(infeed.initializer)
      result, = sess.run(compiled_net)
      report.parse_log()

      # Sanity check results.
      expected_result = np.dot(data, data)
      expected_result = np.dot(expected_result, expected_result)
      self.assertAllEqual(expected_result, result)

      matmuls = []
      iics = []
      for t in report.get_tensor_map().all_tensors():
        # Find matmul operations.
        if t.id.startswith("dot"):
          matmuls.append(t)
        # Find ipu-inter-copy operations.
        elif t.id.startswith("ipu-inter-copy"):
          iics.append(t)

      self.assertEqual(len(matmuls), 2)
      self.assertEqual(len(iics), 1)
      iic = iics[0]

      # Copy to device without I/O tiles should be mapped to more tiles than
      # the source tensor from a device with I/O tiles.
      self.assertGreater(len(iic.tiles), len(matmuls[0].tiles))
      self.assertGreater(len(iic.tiles), num_compute_tiles)
      self.assertLessEqual(len(iic.tiles), tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def testCopyFromDeviceWithoutIoTilesToDeviceWithIoTiles(self):
    tiles_per_ipu = 1216
    num_io_tiles = 32
    num_compute_tiles = tiles_per_ipu - num_io_tiles

    data = np.ones((tiles_per_ipu, tiles_per_ipu), dtype=np.float32)
    a = array_ops.placeholder(np.float32, data.shape)
    outfeed = ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net(a):
      with ipu_shard(0):
        with ops.name_scope("matmul1"):
          a = math_ops.matmul(a, a)
      with ipu_shard(1):
        with ops.name_scope("matmul2"):
          a = math_ops.matmul(a, a)
        with ops.name_scope("outfeed"):
          return outfeed.enqueue(a)

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[a])

    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = tiles_per_ipu
    cfg.io_tiles.num_io_tiles = num_io_tiles
    cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    with session.Session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(compiled_net, {a: data})
      result, = sess.run(outfeed.dequeue())
      report.parse_log()

      # Sanity check results.
      expected_result = np.dot(data, data)
      expected_result = np.dot(expected_result, expected_result)
      self.assertAllEqual(expected_result, result)

      matmuls = []
      iics = []
      for t in report.get_tensor_map().all_tensors():
        # Find matmul operations.
        if t.id.startswith("dot"):
          matmuls.append(t)
        # Find ipu-inter-copy operations.
        elif t.id.startswith("ipu-inter-copy"):
          iics.append(t)

      self.assertEqual(len(matmuls), 2)
      self.assertEqual(len(iics), 1)
      iic = iics[0]

      # Copy to device with I/O tiles should be mapped to fewer tiles than the
      # source tensor from a device without I/O tiles.
      self.assertLess(len(iic.tiles), len(matmuls[0].tiles))
      self.assertLess(len(iic.tiles), num_compute_tiles)


if __name__ == "__main__":
  test.main()

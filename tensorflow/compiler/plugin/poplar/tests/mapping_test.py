# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class MappingTest(xla_test.XLATestCase):
  def testGather(self):
    cfg = ipu.utils.IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_net(w, i):
        out = array_ops.gather(w, i)
        return [out]

      with ops.device('cpu'):
        i = array_ops.placeholder(np.int32, [256])
        w = array_ops.placeholder(np.float32, [1024, 8])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      i_h = np.arange(0, 3 * 256, 3)
      w_h = np.arange(8192).reshape(1024, 8)
      expect = np.take(w_h, i_h, axis=0)

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], expect)

      report_json.parse_log()
      tm = report_json.get_tensor_map()

      bad_maps = []
      for tensor in tm.all_tensors():
        if tensor.num_elements > 16:
          if len(tensor.tiles) == 1 and tensor.has_contant:
            bad_maps += [tensor.inst]

      self.assertFalse(bad_maps)

  def testMappingJson(self):
    cfg = ipu.utils.IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.auto_select_ipus = 1
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.ipu_model.compile_ipu_code = True
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_net(a, b, c):
        a = array_ops.broadcast_to(a, shape=[1024])
        b = array_ops.strided_slice(b, [0], [8192], [8])
        c = array_ops.pad(c, paddings=[[256, 256]])
        out = a + b + c
        return [out]

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [])
        b = array_ops.placeholder(np.float32, [8192])
        c = array_ops.placeholder(np.float32, [512])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[a, b, c])

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      fd = {a: 1.0, b: np.ones([8192]), c: np.ones([512])}
      result = sess.run(r, fd)

      expected = [2] * 256 + [3] * 512 + [2] * 256
      self.assertAllClose(result[0], expected)

      report_json.parse_log()
      tm = report_json.get_tensor_map()

      concat_layout = None
      add_layout = None
      for tensor in tm.all_tensors():
        if tensor.inst.startswith('concatenate'):
          concat_layout = tensor
        elif tensor.inst.startswith('add'):
          add_layout = tensor

      # The concatenated tensor contains 4 elements on 256 tiles
      self.assertEqual(len(concat_layout.tiles), 256)
      for tile_idx, tile in enumerate(concat_layout.tiles):
        self.assertEqual(tile.tile, tile_idx)
        self.assertEqual(tile.num_elements, 4)

      self.assertEqual(len(add_layout.tiles), 128)
      for tile_idx, tile in enumerate(add_layout.tiles):
        self.assertEqual(tile.tile, tile_idx + 64)
        self.assertEqual(tile.num_elements, 4)

      # The add is optimised, inplace and has half of the total tiles used
      self.assertEqual(concat_layout.tiles[64:-64], add_layout.tiles)

  def testInplaceReadWrite(self):
    cfg = ipu.utils.IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_net(x, y, a):
        z = x + y
        c = a + x
        return c, z

      with ops.device('cpu'):
        x = array_ops.placeholder(np.int32, [100])
        y = array_ops.placeholder(np.int32, [100])
        a = array_ops.placeholder(np.int32, [100])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[x, y, a])

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      i_x = np.full(100, 1)
      i_y = np.full(100, 2)
      i_a = np.full(100, 10)
      expect_c = np.full(100, 11)
      expect_z = np.full(100, 3)

      result_c, result_z = sess.run(r, {x: i_x, y: i_y, a: i_a})
      self.assertAllClose(result_c, expect_c)
      self.assertAllClose(result_z, expect_z)

      report_json.parse_log()
      tm = report_json.get_tensor_map()

      bad_maps = []
      for tensor in tm.all_tensors():
        # Number of elements in tensor 100.
        # Number of used tiles should be larger than 1
        if tensor.num_elements != 100 or len(tensor.tiles) <= 1:
          bad_maps += [tensor.inst]

      self.assertFalse(bad_maps)


if __name__ == "__main__":
  googletest.main()

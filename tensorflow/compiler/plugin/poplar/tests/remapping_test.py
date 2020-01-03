# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class MappingTest(xla_test.XLATestCase):
  def testRemap(self):
    with self.session() as sess:

      def my_net(w, i):
        w = ipu.ops.internal_ops.remap(w)
        i = ipu.ops.internal_ops.remap(i)
        out = array_ops.gather(w, i)
        return [out]

      with ops.device('cpu'):
        i = array_ops.placeholder(np.int32, [8])
        w = array_ops.placeholder(np.float32, [32 * 1024])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

      report = ReportJSON(self, sess)
      report.reset()

      i_h = np.arange(0, 8)
      w_h = np.arange(32 * 1024)

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h))

      report.parse_log()
      tm = report.get_tensor_map()

      bad_maps = []
      for tensor in tm.all_tensors():
        # Total elements > 16
        if tensor.num_elements > 16:
          # Tiles used != 1024
          if len(tensor.tiles) != 1024:
            bad_maps += [tensor.inst]

      self.assertFalse(bad_maps)

  def testRemapDeduce(self):
    with self.session() as sess:

      def my_net(w, i):
        w = ipu.ops.internal_ops.remap_deduce(w)
        i = ipu.ops.internal_ops.remap_deduce(i)
        out = array_ops.gather(w, i)
        return [out]

      with ops.device('cpu'):
        i = array_ops.placeholder(np.int32, [8])
        w = array_ops.placeholder(np.float32, [32, 1024])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

      report = ReportJSON(self, sess)
      report.reset()

      i_h = np.arange(2, 10)
      w_h = np.reshape(np.arange(32 * 1024), [32, 1024])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], w_h[2:10])

      report.parse_log()
      tm = report.get_tensor_map()

      bad_maps = []
      for tensor in tm.all_tensors():
        # Total elements > 16
        if tensor.num_elements > 16:
          for tile in tensor.tiles:
            if tile.num_elements > 32:
              bad_maps += [tensor.inst]
      self.assertFalse(bad_maps)


if __name__ == "__main__":
  googletest.main()

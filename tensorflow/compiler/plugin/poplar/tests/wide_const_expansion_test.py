from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op


class WideConstExpansionTest(xla_test.XLATestCase):
  def testCheckMaxTileSize(self):
    with self.session() as sess:
      dtype = np.float32
      shape = (1024, 2048)
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = variable_scope.get_variable(
              "a",
              shape=shape,
              initializer=init_ops.constant_initializer(2),
              dtype=dtype)
        pb = array_ops.placeholder(shape=shape, dtype=dtype, name="b")
        c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
        output = a + pb + c

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())

      report.parse_log()
      report.assert_max_tile_memory_in_range(10000, 12000)

      out = sess.run(output, {pb: np.ones(shape=shape, dtype=dtype)})
      self.assertAllClose(np.full(shape, 7, dtype=dtype), out)

      report.parse_log()
      report.assert_max_tile_memory_in_range(30000, 33000)

  def testWideConstantWithAllocationTarget(self):
    with self.session() as sess:
      # This test will fail if the dynamic slice is not mapped correctly.
      dtype = np.float32
      shape = (512, 2, 2048)

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 2

        def body(i, x, y):
          s = array_ops.slice(x, [i, i, i], [1, 1, 2048])
          y = y + math_ops.reduce_mean(s)
          x = x + constant_op.constant(1, shape=shape, dtype=dtype)
          i = i + 1
          return (i, x, y)

        i = 0
        c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
        return control_flow_ops.while_loop(cond, body, (i, c, y), name='')[2]

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      report = ReportJSON(self, sess, io_trace=False)
      report.reset()

      y = sess.run(r, {y: [10]})
      self.assertAllClose(y[0], [19])

      report.parse_log(assert_len=3)

      ok = [
          '__seed*', 'Copy_*_to_*', 'Slice/dynamic-slice*/dynamicSlice',
          'Mean/reduce', 'Mean/multiply', 'add*/add*/AddTo',
          'add_*/fusion/Op/Add'
      ]
      report.assert_all_compute_sets_and_list(ok)

      report.assert_max_tile_memory_in_range(10000, 15000)
      report.assert_always_live_memory_in_range(10000, 15000)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

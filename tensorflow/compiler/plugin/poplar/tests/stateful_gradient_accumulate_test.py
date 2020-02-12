from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ipu import utils


class StatefulGradientAccumulateTest(xla_test.XLATestCase):
  def testStatefulGradientAccumulate(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=5, verify_usage=False)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [20])
      self.assertAllEqual(y[2], [20])

  def testStatefulGradientAccumulateInvalidUse(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=5)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "The while/IpuStatefulGradientAccumulate op"):
        sess.run(r, {y: [10]})

  def testLoopRepeatCountDoesntDivide(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=4, verify_usage=False)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Detected a gradient accumulation operation with 4 number of mini "
          "batches inside of a loop with 10 iterations."):
        sess.run(r, {y: [10]})


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import ops
from tensorflow.python.ipu import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class ReplicatedStatefulGradientAccumulateTest(xla_test.XLATestCase):
  def testStatefulGradientAccumulateAndCrossReplica(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, y):
          del y
          return i < 10

        def body(i, y):
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(y), num_mini_batches=5)
          cr = gen_popops_ops.ipu_cross_replica_sum(ga)
          y = y + cr
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      opts = utils.auto_select_ipus(opts, num_ipus=2)
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [30])

  def testCrossReplicaAndStatefulGradientAccumulate(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, y):
          del y
          return i < 10

        def body(i, y):
          cr = gen_popops_ops.ipu_cross_replica_sum(array_ops.ones_like(y))
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              cr, num_mini_batches=5)
          y = y + ga
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      opts = utils.auto_select_ipus(opts, num_ipus=2)
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [30])

  def testCrossReplicaAndNormalizeAndStatefulGradientAccumulate(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, y):
          del y
          return i < 10

        def body(i, y):
          cr = gen_popops_ops.ipu_cross_replica_sum(array_ops.ones_like(y))
          norm = gen_poputil_ops.ipu_replication_normalise(cr)
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              norm, num_mini_batches=5)
          y = y + ga
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = utils.create_ipu_config()
      opts = utils.auto_select_ipus(opts, num_ipus=2)
      utils.configure_ipu_system(opts)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [20])


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

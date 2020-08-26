#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import os
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.compiler.xla import xla


class IpuGatherLookupTest(xla_test.XLATestCase, parameterized.TestCase):
  # Overriding abstract cached_session.
  def cached_session(self):
    return 0

  def test_session(self):
    return 0

  @parameterized.parameters(range(1, 10))
  def testGatherLookupRandomize(self, y_0):
    # Configure argument for targeting the IPU.
    # gather_simplifier is on.
    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    self.assertFalse(cfg.enable_gather_simplifier)
    cfg = utils.set_optimization_options(cfg, gather_simplifier=True)
    self.assertTrue(cfg.enable_gather_simplifier)
    utils.configure_ipu_system(cfg)

    # Set test range shape.
    w_0 = 5
    w_1 = 10

    def network(w, y):
      g = nn.embedding_lookup(w, y)
      return g

    # Compare cpu gather vs ipu gather_simplifier.
    with self.session() as sess:
      with ops.device('cpu'):
        y = array_ops.placeholder(np.int32, shape=[y_0])
        w = array_ops.placeholder(np.int32, shape=[w_0, w_1])
        y_i = np.random.randint(low=0, high=w_0 - 1, size=y_0)
        w_i = np.reshape(np.random.randint(low=100, high=200, size=w_0 * w_1),
                         (w_0, w_1))
        cpu_take = array_ops.gather(w_i, y_i)

        report = tu.ReportJSON(self, sess=sess, configure_device=False)

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[w, y])

      sess.run(variables.global_variables_initializer())
      report.reset()
      ipu_gather_simplifier = sess.run(r, {y: y_i, w: w_i})
      self.assertAllClose(ipu_gather_simplifier[0], cpu_take)

      report.parse_log()
      # pylint: disable=line-too-long

      # This tests gather simplifier hlo pass for embedding_lookup case.
      # It checks if "embedding_lookup/gather*/multiSlice" string was
      # replaced by embedding_lookup/multi-slice/*/multiSlice".
      ok = [
          'embedding_lookup/multi-slice/output/multiSlice/*',
          '__seed/set/setMasterSeed',
          'host-exchange-local-copy-',
      ]
      if y_0 == 1:
        ok = ok[:-1]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

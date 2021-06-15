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
# ==============================================================================
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


@test_util.deprecated_graph_mode_only  # pylint: disable=abstract-method
class TestClearXlaCompilationCache(xla_test.XLATestCase):
  def setUp(self):
    xla_test.XLATestCase.setUp(self)

    def test_net(a):
      return a + a

    with ops.device("/device:IPU:0"):
      placeholder_input = array_ops.placeholder(np.float32, shape=[1])
      self.outputs_ = [
          ipu.ipu_compiler.compile(test_net, inputs=[placeholder_input])
      ]

    self.feed_dict_ = {placeholder_input: [1.0]}

  def testCachedCompilationByDefault(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with session.Session() as sess:
      sess.run(self.outputs_, self.feed_dict_)
      sess.run(self.outputs_, self.feed_dict_)
      sess.run(self.outputs_, self.feed_dict_)

      report_helper.assert_num_reports(1)

  def testClearingCausesRecompilation(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with session.Session() as sess:
      sess.run(self.outputs_, self.feed_dict_)
      report_helper.assert_num_reports(1)

      report_helper.clear_reports()
      sess.run([gen_ipu_ops.ipu_clear_all_xla_compilation_caches()])
      sess.run(self.outputs_, self.feed_dict_)
      report_helper.assert_num_reports(1)

      report_helper.clear_reports()
      sess.run([gen_ipu_ops.ipu_clear_all_xla_compilation_caches()])
      sess.run(self.outputs_, self.feed_dict_)
      report_helper.assert_num_reports(1)

  def testClearingEmptyCacheIsSafe(self):
    with session.Session() as sess:

      try:
        sess.run([gen_ipu_ops.ipu_clear_all_xla_compilation_caches()])
      except Exception:  # pylint: disable=broad-except
        self.fail("Clearing an empty cache threw an unexpected exception")


if __name__ == "__main__":
  googletest.main()

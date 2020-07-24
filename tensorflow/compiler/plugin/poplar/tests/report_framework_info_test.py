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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

from tensorflow.python.ipu.scopes import ipu_scope


def is_json(test_str):
  test_s = json.dumps(test_str)
  try:
    json_object = json.loads(test_s)
  except ValueError:
    return False, json_object
  return True, json_object


# pylint: disable=abstract-method
class AutoReportDirTest(xla_test.XLATestCase):
  def testReportInfoDirCreated0(self):
    with self.session() as sess:
      tmpdir = tempfile.mkdtemp()
      cfg = ipu.utils.create_ipu_config(profiling=True,
                                        use_poplar_text_report=True,
                                        profile_execution=True,
                                        report_directory=tmpdir)
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      with ops.device("cpu"):
        pa = array_ops.placeholder(np.float32, [2], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      def basic_graph(pa, pb, pc):
        o1 = pa + pb
        o2 = pa + pc
        simple_graph_output = o1 + o2
        return simple_graph_output

      with ipu_scope("/device:IPU:0"):
        result = basic_graph(pa, pb, pc)

        result = sess.run(result,
                          feed_dict={
                              pa: [1., 1.],
                              pb: [0., 1.],
                              pc: [1., 5.]
                          })

        tmpdir_files = os.listdir(tmpdir)
        self.assertEqual(1, len(tmpdir_files))
        tmpdir_sub = tmpdir + "/" + tmpdir_files[0]
        tmpdir_sub_files = os.listdir(tmpdir_sub)
        self.assertTrue("framework.json" in tmpdir_sub_files)
        test_file = tmpdir_sub + "/framework.json"
        with open(test_file) as f:
          json_txt = json.load(f)
          is_j, _ = is_json(json_txt)
          self.assertTrue(is_j)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

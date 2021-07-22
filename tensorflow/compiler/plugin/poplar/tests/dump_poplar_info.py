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

import glob
import os
import tempfile
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class DumpPoplarInfo(xla_test.XLATestCase):
  def testVertexGraphAndIntervalReport(self):
    with self.session() as sess:

      tempdir = tempfile.mkdtemp('report_dir')
      os.environ['TF_POPLAR_FLAGS'] = ('--save_vertex_graph=' + tempdir + " " +
                                       '--save_interval_report=' + tempdir +
                                       " " +
                                       os.environ.get('TF_POPLAR_FLAGS', ''))

      def my_model(pa, pb, pc):
        output = pa + pb + pc
        return [output]

      with ops.device("cpu"):
        pa = array_ops.placeholder(np.float32, [2048], name="a")
        pb = array_ops.placeholder(np.float32, [2048], name="b")
        pc = array_ops.placeholder(np.float32, [2048], name="c")

      with ops.device("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_model, inputs=[pa, pb, pc])

      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.configure_ipu_system()

      fd = {pa: [1.] * 2048, pb: [2.] * 2048, pc: [3.] * 2048}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, [6.] * 2048)

      vertex_graphs = glob.glob(os.path.join(tempdir, "*.vertex_graph"))
      interval_reports = glob.glob(os.path.join(tempdir, "*.csv"))
      self.assertEqual(len(vertex_graphs), 1)
      self.assertEqual(len(interval_reports), 1)


if __name__ == "__main__":
  googletest.main()

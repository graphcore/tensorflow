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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class MappingTest(xla_test.XLATestCase):
  def testConcat(self):
    with self.session() as sess:

      def my_net(a, b, c, d):
        with ipu.scopes.ipu_shard(0):
          c1 = array_ops.concat([a, b, d], axis=0)
          c2 = array_ops.concat([a, b, c], axis=0)
        return [c1, c2]

      with ops.device('cpu'):
        a = array_ops.placeholder(np.int32, [1])
        b = array_ops.placeholder(np.int32, [1])
        c = array_ops.placeholder(np.int32, [1])
        d = array_ops.placeholder(np.int32, [1])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[a, b, c, d])

      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.configure_ipu_system()

      result = sess.run(r, {a: [0], b: [1], c: [2], d: [3]})
      self.assertAllClose(result[0], [0, 1, 3])
      self.assertAllClose(result[1], [0, 1, 2])


if __name__ == "__main__":
  googletest.main()

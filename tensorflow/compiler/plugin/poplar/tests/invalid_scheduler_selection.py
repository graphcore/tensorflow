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
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class InvalidSchedulerSelectionTest(xla_test.XLATestCase):
  def testInvalidSchedulerSelection(self):
    with self.session() as sess:

      def my_net(a, b):
        out = a + b
        return [out]

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [1])
        b = array_ops.placeholder(np.float32, [1])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[a, b])

      cfg = ipu.utils.create_ipu_config(profiling=True,
                                        scheduler_selection="invalid")
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      ipu.utils.configure_ipu_system(cfg)

      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(r, {a: [1.0], b: [2.0]})


if __name__ == "__main__":
  googletest.main()

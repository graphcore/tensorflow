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

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class EmbeddingLookupTest(test_util.TensorFlowTestCase):
  def testGather(self):
    def my_net(w, i):
      out = ipu.ops.embedding_ops.embedding_lookup(w, i, min_encoding_size=1200)
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8])
      w = array_ops.placeholder(np.float32, [12000, 200])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, 8)
      w_h = np.arange(2400000).reshape([12000, 200])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))


if __name__ == "__main__":
  googletest.main()

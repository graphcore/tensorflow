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
# ==============================================================================

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class OutfeedOrdinalTest(test_util.TensorFlowTestCase):
  def testTwoOutfeedsOnDifferentOrdinals(self):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = [1, 1]
    cfg.configure_ipu_system()

    outfeed_1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(device_ordinal=0)
    outfeed_2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(device_ordinal=1)

    def graph1(tensor):
      enqueue_op = outfeed_1.enqueue(tensor)
      return enqueue_op

    def graph2(tensor):
      enqueue_op = outfeed_2.enqueue(tensor)
      return enqueue_op

    with ops.device('cpu'):
      v1 = array_ops.placeholder(np.float32, [1, 3])
      v2 = array_ops.placeholder(np.float32, [1, 4])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res1 = ipu.ipu_compiler.compile(graph1, inputs=[v1])
    with ipu.scopes.ipu_scope('/device:IPU:1'):
      res2 = ipu.ipu_compiler.compile(graph2, inputs=[v2])

    dequeue_outfeed_ops = []
    dequeue_outfeed_1 = outfeed_1.dequeue()
    dequeue_outfeed_2 = outfeed_2.dequeue()
    dequeue_outfeed_ops.append(dequeue_outfeed_1)
    dequeue_outfeed_ops.append(dequeue_outfeed_2)

    with session_lib.Session() as sess:
      sess.run(res1, {v1: np.ones([1, 3])})
      sess.run(res2, {v2: np.ones([1, 4])})
      results = sess.run(dequeue_outfeed_ops)
      self.assertAllClose(results[0], np.broadcast_to(1, [1, 1, 3]))
      self.assertAllClose(results[1], np.broadcast_to(1, [1, 1, 4]))


if __name__ == "__main__":
  googletest.main()

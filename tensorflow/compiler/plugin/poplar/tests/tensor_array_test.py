# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops


class IpuXlaTensorArrayTest(xla_test.XLATestCase):
  def testTensorArrayWriteRead(self):
    with ops.device("/device:IPU:0"):
      with self.session() as session:

        in1 = array_ops.placeholder(np.float32, [1, 2])
        in2 = array_ops.placeholder(np.float32, [1, 2])
        in3 = array_ops.placeholder(np.float32, [1, 2])

        ta = tensor_array_ops.TensorArray(dtype=np.float32,
                                          tensor_array_name="foo",
                                          size=3)

        w0 = ta.write(0, in1)
        w1 = w0.write(1, in2)
        w2 = w1.write(2, in3)

        r0 = w2.read(0)
        r1 = w2.read(1)
        r2 = w2.read(2)

        d0, d1, d2 = session.run([r0, r1, r2],
                                 feed_dict={
                                     in1: [[4.0, 5.0]],
                                     in2: [[1.0, 3.0]],
                                     in3: [[7.0, -8.5]]
                                 })
        self.assertAllEqual([[4.0, 5.0]], d0)
        self.assertAllEqual([[1.0, 3.0]], d1)
        self.assertAllEqual([[7.0, -8.5]], d2)

  def testTensorArrayScatterGather(self):
    with ops.device("/device:IPU:0"):
      with self.session() as session:

        in1 = array_ops.placeholder(np.float32, [5, 2])
        in2 = array_ops.placeholder(np.float32, [2])

        ta = tensor_array_ops.TensorArray(dtype=np.float32,
                                          tensor_array_name="ta",
                                          size=5)

        tb = tensor_array_ops.TensorArray(dtype=np.float32,
                                          tensor_array_name="tb",
                                          size=5)

        ta = ta.unstack(in1)
        tb = tb.write(0, ta.read(0) + in2)
        tb = tb.write(1, ta.read(1) + in2)
        tb = tb.write(2, ta.read(2) + in2)
        tb = tb.write(3, ta.read(3) + in2)
        tb = tb.write(4, ta.read(4) + in2)
        out = tb.gather(list(range(4)))

        v = session.run(out,
                        feed_dict={
                            in1: [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                            in2: [1, 1]
                        })

        self.assertAllEqual([[2, 2], [3, 3], [4, 4], [5, 5]], v)


if __name__ == "__main__":
  googletest.main()

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

from tensorflow.python.client import session
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest


#TODO Test with a replicated graph
class ContribIpuOpsTest(test_util.TensorFlowTestCase):
  def testCrossReplicaSumOp(self):
    with ops.device("/device:IPU:0"):
      with session.Session() as s:
        t1 = random_ops.random_uniform([1000], dtype=dtypes.float32)
        t2 = ipu.ops.cross_replica_ops.cross_replica_sum(t1, name="crs")
        h1, h2 = s.run([t1, t2])
        self.assertEqual(list(h1), list(h2))


if __name__ == "__main__":
  googletest.main()

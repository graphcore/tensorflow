# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


class TensorPathTransformTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def testAllocationPathWithReduce(self):
    with self.session() as session:
      with ops.device('cpu'):
        indices = array_ops.placeholder(np.int32, [8])
        sequence = array_ops.placeholder(np.float32, [32, 16])

      def my_net(indices, sequence):
        # Essentially multiply indices by 4 (n * 3 + n).
        indices = math_ops.multiply(indices, 3)
        const = array_ops.constant(list(range(8)), np.int32)
        const = array_ops.expand_dims(const, axis=1)
        indices = array_ops.expand_dims(indices, axis=1)
        indices = array_ops.concat([indices, const], axis=1)

        # The output shape of this reduce is the same as the original shape of
        # the indices input. This means the indices input will be allocated
        # through this reduce.
        indices = math_ops.reduce_sum(indices, axis=1)
        return ipu.ops.embedding_ops.embedding_lookup(sequence, indices)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        out = ipu.ipu_compiler.compile(my_net, inputs=[indices, sequence])

      indices_data = list(range(8))
      sequence_data = np.reshape(list(range(32 * 16)), [32, 16])
      result, = session.run(out,
                            feed_dict={
                                indices: indices_data,
                                sequence: sequence_data,
                            })

    expected_result = [sequence_data[i * 4] for i in indices_data]
    self.assertAllEqual(expected_result, result)


if __name__ == "__main__":
  googletest.main()

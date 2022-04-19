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
from tensorflow.python.ipu import test_utils as tu

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
    cfg = ipu.config.IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device('cpu'):
        indices = array_ops.placeholder(np.int32, [2, 8])
        sequence = array_ops.placeholder(np.float32, [32, 16])

      def my_net(indices, sequence):
        # A roundabout way of increasing all indices by 8.
        # Importantly it involves a reduce back to the original input shape.
        const = array_ops.constant([8], np.int32)
        indices = array_ops.expand_dims(indices, axis=2)
        const = array_ops.broadcast_to(const, [2, 8, 1])
        indices = array_ops.concat([indices, const], axis=2)

        # The output shape of this reduce is the same as the original shape of
        # the indices input. This means the indices input will be allocated
        # through this reduce.
        indices = math_ops.reduce_sum(indices, axis=2)
        return ipu.ops.embedding_ops.embedding_lookup(sequence, indices)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        out = ipu.ipu_compiler.compile(my_net, inputs=[indices, sequence])

      indices_data = np.reshape(list(range(2 * 8)), [2, 8])
      sequence_data = np.reshape(list(range(32 * 16)), [32, 16])

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      result, = sess.run(out,
                         feed_dict={
                             indices: indices_data,
                             sequence: sequence_data,
                         })
      report_json.parse_log()

    # Check that the reduce hasn't been optimised out.
    # This would mean the reduce case never got hit in path transform.
    self.assertTrue(
        any(
            name.startswith("reduce") for name in
            report_json.get_tensor_map().tensor_inst_name_mappings()))

    expected_result = [sequence_data[i + 8] for i in indices_data]
    self.assertAllEqual(expected_result, result)


if __name__ == "__main__":
  googletest.main()

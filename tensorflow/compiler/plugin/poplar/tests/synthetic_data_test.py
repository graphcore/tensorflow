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

import os
import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


class IpuXlaVariableTestSyntheticData(xla_test.XLATestCase):
  def testResourceCountsAreCorrect(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          w1 = variable_scope.get_variable(
              "w1",
              shape=[4, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                           dtype=np.float32)))
          b1 = variable_scope.get_variable(
              "b1",
              shape=[2],
              dtype=np.float32,
              trainable=False,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))
          w2 = variable_scope.get_variable(
              "w2",
              shape=[2, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4]], dtype=np.float32)))
          b2 = variable_scope.get_variable(
              "b2",
              shape=[2],
              dtype=np.float32,
              trainable=False,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[1, 4])
        y = math_ops.matmul(x, w1) + b1
        y = math_ops.matmul(y, w2) + b2

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report_json = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report_json.reset()

      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      report_json.parse_log()
      report_json.assert_host_to_device_event_names([])
      report_json.assert_device_to_host_event_names([])

      # Explicitly fetch the first set of weights and biases
      sess.run([w1, b1])

      report_json.parse_log()
      report_json.assert_host_to_device_event_names([])
      report_json.assert_device_to_host_event_names([])


if __name__ == "__main__":
  os.environ["TF_POPLAR_FLAGS"] = (
      "--use_synthetic_data --use_ipu_model --synthetic_data_initializer=random"
  )
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

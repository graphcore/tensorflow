# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import variables
from tensorflow.python.client import session as sl

from tensorflow.python import ipu


class RNNDropoutTest(test_util.TensorFlowTestCase):
  @staticmethod
  def _test(test, model_fn):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result_dropout = ipu.ipu_compiler.compile(model_fn(0.9))
        result_no_dropout = ipu.ipu_compiler.compile(model_fn(0.0))

      ipu.utils.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      output_dropout = sess.run(result_dropout)[0]
      output_no_dropout = sess.run(result_no_dropout)[0]

    output_t0, output_t1 = (output_no_dropout[:, 0, :],
                            output_no_dropout[:, 1, :])
    # Check that all the outputs are the same without dropout for the same
    # timestep.
    test.assertTrue(np.all(output_t0 == output_t0[0][0]))
    test.assertTrue(np.all(output_t1 == output_t1[0][0]))
    output_t0_val = output_t0[0][0]
    output_t1_val = output_t1[0][0]

    output_t0, output_t1 = output_dropout[:, 0, :], output_dropout[:, 1, :]
    # Normalize the outputs.
    output_t0 = output_t0 / output_t0_val
    output_t1 = output_t1 / output_t1_val
    # Make sure non zero elements are in the same locations between time steps.
    test.assertAllEqual(output_t0.nonzero(), output_t1.nonzero())

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_lstm_dropout(self):
    # The layer is created with recurrent_initializer = zero, so that the
    # the recurrent state won't affect the output. By doing this, we can verify
    # the output and see if the same mask is applied to for each timestep.
    def model_wrapper(dropout):
      def model():
        layer = ipu.layers.LSTM(3,
                                dtype=np.float32,
                                kernel_initializer='ones',
                                recurrent_initializer='zeros',
                                bias_initializer='zeros',
                                dropout=dropout,
                                return_state=False,
                                return_sequences=True,
                                unit_forget_bias=False)
        inputs = constant_op.constant(1.0, shape=(6, 2, 5))
        return layer(inputs, training=True)

      return model

    self._test(self, model_wrapper)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_gru_dropout(self):
    # The layer is created with recurrent_initializer = zero, so that the
    # the recurrent state won't affect the output. By doing this, we can verify
    # the output and see if the same mask is applied to for each timestep.
    def model_wrapper(dropout):
      def model():
        layer = ipu.layers.GRU(3,
                               dtype=np.float32,
                               kernel_initializer='ones',
                               recurrent_initializer='zeros',
                               bias_initializer='zeros',
                               dropout=dropout,
                               return_state=False,
                               return_sequences=True)
        inputs = constant_op.constant(1.0, shape=(6, 2, 5))
        return layer(inputs, training=True)

      return model

    self._test(self, model_wrapper)


if __name__ == "__main__":
  googletest.main()

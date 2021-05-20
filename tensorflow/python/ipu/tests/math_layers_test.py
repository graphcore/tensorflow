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
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python.eager import backprop
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

from tensorflow.python.keras.layers import Dense
from tensorflow.python.ipu.keras.layers import SerialDense

TEST_CASES = ({
    'testcase_name': 'input_columns',
    'input_shape': [8, 16],
    'num_units': 5,
    'serialization_factor': 2,
    'serialization_dimension': 'input_columns',
}, {
    'testcase_name': 'input_rows_kernel_columns',
    'input_shape': [4, 21],
    'num_units': 8,
    'serialization_factor': 3,
    'serialization_dimension': 'input_rows_kernel_columns',
}, {
    'testcase_name': 'kernel_rows',
    'input_shape': [4, 4],
    'num_units': 8,
    'serialization_factor': 2,
    'serialization_dimension': 'kernel_rows',
})


def _getTestCases():
  from copy import deepcopy

  test_cases = list(TEST_CASES)
  # Add test cases with a batch dim for a.
  for case in deepcopy(TEST_CASES):
    case['testcase_name'] += "_batch_a"
    case['input_shape'] = [2] + case['input_shape']
    test_cases.append(case)
  return test_cases


# Note that in this test we expect small numerical differences as serializing
# means that some operations are done in a different order.
class SerialDenseTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(0xDEADBEEF)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.run_v2_only
  def testSerialDense(self, input_shape, num_units, serialization_factor,
                      serialization_dimension):
    input_val = np.random.normal(2.0, 2.0, input_shape)
    kernel_val = np.random.normal(2.0, 2.0, [input_shape[-1], num_units])

    def kernel_init(_shape, **_):
      return kernel_val

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      dense = Dense(num_units, kernel_initializer=kernel_init)(input_val)
      serial_dense = SerialDense(num_units,
                                 serialization_factor,
                                 serialization_dimension,
                                 kernel_initializer=kernel_init)(input_val)
      self.assertAllClose(dense, serial_dense, atol=1.e-05, rtol=1.e-05)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.run_v2_only
  def testSerializedMatmulGrad(self, input_shape, num_units,
                               serialization_factor, serialization_dimension):
    input_val = np.random.normal(2.0, 2.0, input_shape)
    kernel_val = np.random.normal(2.0, 2.0, [input_shape[-1], num_units])

    def kernel_init(_shape, **_):
      return kernel_val

    def func(layer):
      with backprop.GradientTape() as t:
        output = layer(input_val)
        # Not a real loss function, but good enough for testing backprop.
        loss = math_ops.reduce_sum(output)
      grads = t.gradient(loss, layer.weights)
      return grads

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      dense = Dense(num_units, kernel_initializer=kernel_init)
      serial_dense = SerialDense(num_units,
                                 serialization_factor,
                                 serialization_dimension,
                                 kernel_initializer=kernel_init)

      out_dense = func(dense)
      out_serial_dense = func(serial_dense)
    self.assertAllClose(out_dense, out_serial_dense, atol=1.e-05, rtol=1.e-05)


if __name__ == "__main__":
  googletest.main()

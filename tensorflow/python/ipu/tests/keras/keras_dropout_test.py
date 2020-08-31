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
"""Test for IPU Dropout layer."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

dataType = np.float32


def _kerasIPUDropout(instance, x_val, rate=0.5, seed=None, training=True):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    output = ipu.layers.Dropout(dtype=dataType, rate=rate,
                                seed=seed)(inputs=x, training=training)

  with instance.test_session() as sess:
    return sess.run(output, {x: x_val})


class IPUDropoutTest(test.TestCase):
  @test_util.deprecated_graph_mode_only
  def testDropout(self):
    np.random.seed(42)

    num_elements = 1000
    x = np.random.rand((num_elements)).astype(dataType)

    # Test rates
    rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.99]
    for r in rates:
      keras_result = _kerasIPUDropout(self, x, r)
      num_non_zero = np.count_nonzero(keras_result)
      percent_drop = num_non_zero / num_elements
      self.assertAllClose(1 - percent_drop, r, rtol=0.05)

    # Test user seed
    num_elements = 100
    x = np.random.rand((num_elements)).astype(dataType)

    keras_result_a = _kerasIPUDropout(self, x, seed=[42, 42])
    for _ in range(0, 6):
      keras_result_b = _kerasIPUDropout(self, x, seed=[42, 42])
      self.assertAllEqual(keras_result_b, keras_result_a)

    # Test scaling of kept elements
    num_elements = 1000
    x = np.ones((num_elements)).astype(dataType)
    for r in rates:
      keras_result = _kerasIPUDropout(self, x, seed=[42, 42], rate=r)
      kept_values = keras_result[np.nonzero(keras_result)]
      expected_kept_values = 1 / (1 - r) * np.ones(kept_values.shape)
      self.assertAllClose(kept_values, expected_kept_values)

  def testInference(self):
    num_elements = 100
    x = np.random.rand((num_elements)).astype(dataType)

    inf_output = _kerasIPUDropout(self, x, seed=[42, 42], training=False)
    self.assertAllClose(inf_output, x)

  def testSingleOutputFromLayer(self):
    num_elements = 100
    x = np.random.rand((num_elements)).astype(dataType)

    output = _kerasIPUDropout(self, x, seed=[42, 42], training=True)
    self.assertTrue(isinstance(output, (np.ndarray)))

  def testIllegalRate(self):
    with self.assertRaisesRegex(
        ValueError, r"The rate must be in the range \[0, 1\), but was 1"):
      _kerasIPUDropout(self, np.ones(1), rate=1)


if __name__ == '__main__':
  test.main()

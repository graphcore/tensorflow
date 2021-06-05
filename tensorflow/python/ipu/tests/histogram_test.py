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
# =============================================================================

import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python.ipu.ops.statistics_ops import histogram
from tensorflow.python.ipu.ops.statistics_ops import histogram_update
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

TEST_CASES = [{
    'testcase_name':
    'AllPositiveF16',
    'inputs':
    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             dtype=np.float16),
    'levels':
    np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float16),
    'absolute_of_input':
    False,
    'expected_distribution':
    np.array([2, 2, 3, 2, 1], dtype=np.int32)
}, {
    'testcase_name':
    'AllPositiveZeroTenF16',
    'inputs':
    np.array([0, 0, 0, 1, 2, 5, 6, 6, 0, 10], dtype=np.float16),
    'levels':
    np.array([5, 10], dtype=np.float16),
    'absolute_of_input':
    False,
    'expected_distribution':
    np.array([6, 3, 1], dtype=np.int32)
}, {
    'testcase_name': 'AllZeroF16',
    'inputs': np.array([0, 0, 0], dtype=np.float16),
    'levels': np.array([-1, 1], dtype=np.float16),
    'absolute_of_input': False,
    'expected_distribution': np.array([0, 3, 0], dtype=np.int32)
}, {
    'testcase_name':
    'PositiveNegativeAbsOfInputF16',
    'inputs':
    np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0],
             dtype=np.float16),
    'levels':
    np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float16),
    'absolute_of_input':
    True,
    'expected_distribution':
    np.array([2, 2, 3, 2, 1], dtype=np.int32)
}, {
    'testcase_name':
    'AllPositiveF32',
    'inputs':
    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             dtype=np.float32),
    'levels':
    np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    'absolute_of_input':
    False,
    'expected_distribution':
    np.array([2, 2, 3, 2, 1], dtype=np.int32)
}, {
    'testcase_name':
    'AllPositiveZeroTenF32',
    'inputs':
    np.array([0, 0, 0, 1, 2, 5, 6, 6, 0, 10], dtype=np.float32),
    'levels':
    np.array([5, 10], dtype=np.float32),
    'absolute_of_input':
    False,
    'expected_distribution':
    np.array([6, 3, 1], dtype=np.int32)
}, {
    'testcase_name': 'AllZeroF32',
    'inputs': np.array([0, 0, 0], dtype=np.float32),
    'levels': np.array([-1, 1], dtype=np.float32),
    'absolute_of_input': False,
    'expected_distribution': np.array([0, 3, 0], dtype=np.int32)
}, {
    'testcase_name':
    'PositiveNegativeAbsOfInputF32',
    'inputs':
    np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0],
             dtype=np.float32),
    'levels':
    np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    'absolute_of_input':
    True,
    'expected_distribution':
    np.array([2, 2, 3, 2, 1], dtype=np.int32)
}]


@test_util.deprecated_graph_mode_only
class PopOpsHistogramTest(test_util.TensorFlowTestCase,
                          parameterized.TestCase):
  def testInvalidInputDtype(self):
    a = np.ones((3,), dtype=np.int32)
    b = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "Only float16 and float32 types are supported"):
      _ = histogram(a, b)

  def testInvalidLevelsDtype(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((3,), dtype=np.int32)

    with self.assertRaisesRegex(
        ValueError, "Only float16 and float32 types are supported"):
      _ = histogram(a, b)

  def testInvalidInputRank(self):
    a = np.ones((1, 3), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(ValueError,
                                "histogram expects rank-1 tensor inputs."):
      _ = histogram(a, b)

  def testInvalidLevelsRank(self):
    a = np.ones((3), dtype=np.float32)
    b = np.ones((1, 3), dtype=np.float32)

    with self.assertRaisesRegex(ValueError,
                                "histogram expects rank-1 tensor inputs."):
      _ = histogram(a, b)

  def testInvalidHistDtypeUpdate(self):
    a = np.ones((3,), dtype=np.int32)
    b = np.ones((3,), dtype=np.float32)
    c = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(ValueError, "hist must be of float32 type"):
      _ = histogram_update(a, b, c)

  def testInvalidInputDtypeUpdate(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((3,), dtype=np.int32)
    c = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "Only float16 and float32 types are supported"):
      _ = histogram_update(a, b, c)

  def testInvalidLevelsDtypeUpdate(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)
    c = np.ones((3,), dtype=np.int32)

    with self.assertRaisesRegex(
        ValueError, "Only float16 and float32 types are supported"):
      _ = histogram_update(a, b, c)

  def testInvalidHistRankUpdate(self):
    a = np.ones((1, 3), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)
    c = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "histogram_update expects rank-1 tensor inputs."):
      _ = histogram_update(a, b, c)

  def testInvalidInputRankUpdate(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((1, 3), dtype=np.float32)
    c = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "histogram_update expects rank-1 tensor inputs."):
      _ = histogram_update(a, b, c)

  def testInvalidLevelsRankUpdate(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)
    c = np.ones((1, 3), dtype=np.float32)

    with self.assertRaisesRegex(
        ValueError, "histogram_update expects rank-1 tensor inputs."):
      _ = histogram_update(a, b, c)

  def testHistLevelsShapeMismatchUpdate(self):
    a = np.ones((3,), dtype=np.float32)
    b = np.ones((3,), dtype=np.float32)
    c = np.ones((3,), dtype=np.float32)

    with self.assertRaisesRegex(ValueError,
                                "hist and levels shapes are incompatible"):
      _ = histogram_update(a, b, c)

  @parameterized.named_parameters(*TEST_CASES)
  def testMakeHistogram(self, inputs, levels, absolute_of_input,
                        expected_distribution):
    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        inputs_ph = array_ops.placeholder(inputs.dtype, inputs.shape)
        levels_ph = array_ops.placeholder(levels.dtype, levels.shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def f(x, l):
          return histogram(x, l, absolute_of_input)

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[inputs_ph, levels_ph])

        feed_dict = {inputs_ph: inputs, levels_ph: levels}

        res = sess.run(f_compiled, feed_dict)

    self.assertAllEqual(res[0], expected_distribution)

  @parameterized.named_parameters(*TEST_CASES)
  def testUpdateHistogram(self, inputs, levels, absolute_of_input,
                          expected_distribution):
    with sl.Session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device('cpu'):
        inputs_ph = array_ops.placeholder(inputs.dtype, inputs.shape)
        levels_ph = array_ops.placeholder(levels.dtype, levels.shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):

        def f(x, l):
          hist = histogram(x, l, absolute_of_input)
          hist = histogram_update(hist, x, l, absolute_of_input)
          return histogram_update(hist, x, l, absolute_of_input)

        f_compiled = ipu.ipu_compiler.compile(f, inputs=[inputs_ph, levels_ph])

        feed_dict = {inputs_ph: inputs, levels_ph: levels}

        res = sess.run(f_compiled, feed_dict)

    self.assertAllEqual(res[0], 3.0 * expected_distribution)


if __name__ == "__main__":
  googletest.main()

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import numpy as np
from test_utils import ReportJSON

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

TYPES = (np.float16, np.float32, np.int32)
TESTCASES = [{"testcase_name": np.dtype(x).name, "dtype": x} for x in TYPES]


def _get_random_input(dtype, shape):
  if np.issubdtype(dtype, np.integer):
    info_fn = np.iinfo
    random_fn = np.random.random_integers
  else:
    info_fn = np.finfo
    random_fn = np.random.uniform
  return random_fn(info_fn(dtype).min, info_fn(dtype).max,
                   size=shape).astype(dtype)


class ArgMinMax(xla_test.XLATestCase, parameterized.TestCase):
  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxBasic(self, dtype):
    cfg = IPUConfig()
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with self.session() as sess:
      report_json = ReportJSON(self, sess)
      report_json.reset()

      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3, 5, 2])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=0))

      report_json.parse_log(assert_len=4)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxHalf(self, dtype):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmax(a, output_type=dtypes.int32)

    with self.session() as sess:
      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3, 5, 2])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=0))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxMultiDimensional(self, dtype):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a, axis):
      return math_ops.argmax(a, axis=axis, output_type=dtypes.int32)

    for axis in range(6):
      with self.session() as sess:
        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
          p_axis = array_ops.placeholder(np.int32, shape=())

        with ops.device("/device:IPU:0"):
          out = model(pa, p_axis)

        input = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmax(input, axis=axis))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinBasic(self, dtype):
    cfg = IPUConfig()
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with self.session() as sess:
      report_json = ReportJSON(self, sess)

      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3, 5, 2])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      report_json.reset()

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmin(input, axis=0))

      report_json.parse_log(assert_len=4)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinHalf(self, dtype):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmin(a, output_type=dtypes.int32)

    with self.session() as sess:
      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3, 5, 2])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmin(input, axis=0))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMinMultiDimensional(self, dtype):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a, axis):
      return math_ops.argmin(a, axis=axis, output_type=dtypes.int32)

    for axis in range(6):
      with self.session() as sess:
        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [1, 2, 3, 4, 5, 6])
          p_axis = array_ops.placeholder(np.int32, shape=())

        with ops.device("/device:IPU:0"):
          out = model(pa, p_axis)

        input = _get_random_input(dtype, (1, 2, 3, 4, 5, 6))

        fd = {pa: input, p_axis: axis}
        result = sess.run(out, fd)
        self.assertAllClose(result, np.argmin(input, axis=axis))

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxNegativeDim(self, dtype):
    cfg = IPUConfig()
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmax(a, axis=-1, output_type=dtypes.int32)

    with self.session() as sess:
      report_json = ReportJSON(self, sess)
      report_json.reset()

      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3, 5, 2])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      input = _get_random_input(dtype, (3, 5, 2))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=-1))

      report_json.parse_log(assert_len=4)

  @parameterized.named_parameters(*TESTCASES)
  def testArgMaxVector(self, dtype):
    cfg = IPUConfig()
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def model(a):
      return math_ops.argmax(a, axis=0, output_type=dtypes.int32)

    with self.session() as sess:
      report_json = ReportJSON(self, sess)
      report_json.reset()

      with ops.device('cpu'):
        pa = array_ops.placeholder(dtype, [3])

      with ops.device("/device:IPU:0"):
        out = model(pa)

      input = _get_random_input(dtype, (3))

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input))

      report_json.parse_log(assert_len=4)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

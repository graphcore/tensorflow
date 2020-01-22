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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python import ipu


class FunctionTest(test_util.TensorFlowTestCase):
  @test_util.run_v2_only
  def testBasicFunction(self):
    @ipu.function
    def my_func(a, b, c):
      return a + b + c

    r = tu.ReportJSON(self, eager_mode=True)

    result = my_func(constant_op.constant(1, shape=[2]),
                     constant_op.constant(2, shape=[2]),
                     constant_op.constant(3, shape=[2]))
    self.assertAllEqual([6, 6], result.numpy())

    r.parse_log(assert_len=4)
    r.assert_contains_one_compile_event()

    cs = ['add/*/AddTo', 'add_1/*/AddTo', '__seed']
    r.assert_compute_sets_contain_list(cs)

  @test_util.run_v2_only
  def testBasicFunctionWithConstantInputs(self):
    @ipu.function
    def my_func(a, b, c):
      return a + b + c

    r = tu.ReportJSON(self, eager_mode=True)

    result = my_func(constant_op.constant(1, shape=[2]), 2, c=3)
    self.assertAllEqual([6, 6], result.numpy())

    r.parse_log(assert_len=4)
    r.assert_contains_one_compile_event()

    cs = ['add_1/*/Add', '__seed']
    r.assert_compute_sets_contain_list(cs)

  @test_util.run_v2_only
  def testBasicFunctionWithUnsupportedOp(self):
    @ipu.function
    def my_func(a):
      return dct_ops.dct(a)

    tu.ReportJSON(self, eager_mode=True)

    with self.assertRaises(errors.InvalidArgumentError):
      my_func(constant_op.constant([1., 2., 1., 4., 1., 6.]))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

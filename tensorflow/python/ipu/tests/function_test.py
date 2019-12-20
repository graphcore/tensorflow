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

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
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

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    result = my_func(constant_op.constant(1), constant_op.constant(2),
                     constant_op.constant(3))

    self.assertAllEqual(6, result.numpy())

    r = tu.ReportJSON(self)
    rep = gen_ipu_ops.ipu_event_trace()
    types = r.parse_events(rep)

    assert types[IpuTraceEvent.COMPILE_BEGIN] == 1
    assert types[IpuTraceEvent.COMPILE_END] == 1

    cs = ['add/*/AddTo', 'add_1/*/AddTo', '__seed']
    r.assert_compute_sets_contain_list(cs)

  @test_util.run_v2_only
  def testBasicFunctionWithConstantInputs(self):
    @ipu.function
    def my_func(a, b, c):
      return a + b + c

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    result = my_func(constant_op.constant(1), 2, 3)

    self.assertAllEqual(6, result.numpy())

    r = tu.ReportJSON(self)
    rep = gen_ipu_ops.ipu_event_trace()
    types = r.parse_events(rep)

    assert types[IpuTraceEvent.COMPILE_BEGIN] == 1
    assert types[IpuTraceEvent.COMPILE_END] == 1

    cs = ['add_1/*/AddTo', '__seed']
    r.assert_compute_sets_contain_list(cs)

  @test_util.run_v2_only
  def testBasicFunctionWithUnsupportedOp(self):
    @ipu.function
    def my_func(a):
      return dct_ops.dct(a)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with self.assertRaises(errors.InvalidArgumentError):
      my_func(constant_op.constant([1., 2., 1., 4., 1., 6.]))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

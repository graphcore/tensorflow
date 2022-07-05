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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pva

from absl.testing import parameterized

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent

# Batch
# input spatial dim size
# filter spatial dim size
# input chan per group
# output chans per group
# groups
tests = [
    [1, 1, 1, 2, 3, 2],
    [1, 12, 3, 8, 3, 4],
    [1, 12, 3, 64, 9, 16],
]

TEST_CASES = [{
    'testcase_name': "_".join([str(x) for x in c]),
    'B': c[0],
    'D': c[1],
    'F': c[2],
    'C': c[3],
    'N': c[4],
    'G': c[5],
} for c in tests]


def _compare_ipu_to_cpu(test_wrapper,
                        model_fn,
                        inputs_fn,
                        init_values,
                        compute_sets=None,
                        partial_compute_sets=None):
  def _run_on_ipu():
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      g.add_to_collection("run_type", "ipu")
      inputs = inputs_fn()
      fd = dict(zip(inputs, init_values))
      with variable_scope.variable_scope("ipu", use_resource=True,
                                         reuse=False):
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          res = ipu.ipu_compiler.compile(model_fn, inputs=inputs)

      tu.move_variable_initialization_to_cpu()
      session.run(variables.global_variables_initializer())

      session.run(res, fd)

      report = pva.openReport(report_helper.find_report())
      if compute_sets:
        test_wrapper.assert_all_compute_sets_and_list(report, compute_sets)
      if partial_compute_sets:
        test_wrapper.assert_compute_sets_contain_list(report,
                                                      partial_compute_sets)

      tvars = session.run(variables.trainable_variables())
      return tvars

  def _run_on_cpu():
    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      g.add_to_collection("run_type", "cpu")
      inputs = inputs_fn()
      fd = dict(zip(inputs, init_values))
      with variable_scope.variable_scope("cpu", use_resource=True,
                                         reuse=False):
        with ipu.scopes.ipu_scope("/device:XLA_CPU:0"):
          res = model_fn(*inputs)
      with ops.device("cpu"):
        session.run(variables.global_variables_initializer())
        session.run(res, fd)
        tvars = session.run(variables.trainable_variables())
        return tvars

  vars_ipu = _run_on_ipu()
  vars_cpu = _run_on_cpu()

  test_wrapper.assertAllClose(vars_cpu, vars_ipu, 1e-3)


class GroupedConvTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*TEST_CASES)
  @test_util.deprecated_graph_mode_only
  def testGroupedConvolutions(self, B, D, F, C, N, G):

    initial_weights = np.random.random_sample([F, F, C,
                                               N * G]).astype(np.float32)
    initial_bias = np.random.random_sample([C * G]).astype(np.float32)

    def body(inp, lab):
      weights = variable_scope.get_variable("weights",
                                            initializer=initial_weights)
      bias = variable_scope.get_variable("bias", initializer=initial_bias)

      # Do bias first to ensure that there is an input grad on the conv
      a = inp + bias

      a = nn.convolution(a, weights, padding="VALID", name='cnv')

      a = math_ops.reduce_mean(a, axis=[2, 3])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=a,
                                                          labels=lab))

      return gradient_descent.GradientDescentOptimizer(0.001).minimize(loss)

    def inputs_fn():
      with ops.device('cpu'):
        inp = array_ops.placeholder(np.float32, [B, D, D, C * G], name="input")
        lab = array_ops.placeholder(np.int32, [1])
      return inp, lab

    init_values = [np.random.random_sample([B, D, D, C * G]), np.array([0])]

    cs = [
        'ipu/cnv/convolution*/Conv_*/Convolve',
        'ipu/gradients/ipu/cnv_grad/Conv2DBackpropInput/conv-with-reverse.*/*',
        'ipu/gradients/ipu/cnv_grad/Conv2DBackpropFilter/fusion*'
    ]
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        partial_compute_sets=cs)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

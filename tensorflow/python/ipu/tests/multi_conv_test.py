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
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pva

from google.protobuf import json_format

from tensorflow.python.ipu import test_utils as tu
from tensorflow.compiler.plugin.poplar.driver import option_flag_pb2
from tensorflow.python import ipu
from tensorflow.python.keras import layers
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


def _test_multi_conv_wrapper(func):
  # Don't wrap with the scope on the CPU.
  on_cpu = "cpu" in ops.get_default_graph().get_collection("run_type")
  if on_cpu:
    return func

  @ipu.nn_ops.multi_conv
  def func_wrapper(*args):
    return func(*args)

  return func_wrapper


def _compare_ipu_to_cpu(test_wrapper,
                        model_fn,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        compute_sets=None,
                        partial_compute_sets=None):
  def _run_on_ipu():
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.optimizations.math.dot_strength = False
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

      report_json = tu.ReportJSON(test_wrapper, session)
      tu.move_variable_initialization_to_cpu()
      session.run(variables.global_variables_initializer())
      report_json.reset()
      r = session.run(res, fd)[0]

      report_json.parse_log()
      test_wrapper.assertAllEqual(report_json.get_ml_type_counts(),
                                  conv_classifications)

      report = pva.openReport(report_helper.find_report())
      if compute_sets:
        test_wrapper.assert_all_compute_sets_and_list(report, compute_sets)
      if partial_compute_sets:
        test_wrapper.assert_compute_sets_contain_list(report,
                                                      partial_compute_sets)

      tvars = session.run(variables.trainable_variables())
      return r, tvars

  def _run_on_cpu():
    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      g.add_to_collection("run_type", "cpu")
      inputs = inputs_fn()
      fd = dict(zip(inputs, init_values))
      with variable_scope.variable_scope("cpu", use_resource=True,
                                         reuse=False):
        res = model_fn(*inputs)
      with ops.device("cpu"):
        session.run(variables.global_variables_initializer())
        r = session.run(res, fd)[0]
        tvars = session.run(variables.trainable_variables())
        return r, tvars

  res_ipu, vars_ipu = _run_on_ipu()
  res_cpu, vars_cpu = _run_on_cpu()
  # Tolerance taken from compiler conv tests.
  test_wrapper.assertAllClose(res_cpu, res_ipu, 1e-3)
  test_wrapper.assertAllClose(vars_cpu, vars_ipu, 1e-3)


class MultiConvTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testTraining(self):
    def convs(x, y):
      a = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        kernel_initializer=init_ops.ones_initializer(),
                        bias_initializer=init_ops.constant_initializer(0.5),
                        name='conv1')(x)
      b = layers.Conv2D(4,
                        1,
                        use_bias=True,
                        kernel_initializer=init_ops.ones_initializer(),
                        bias_initializer=init_ops.constant_initializer(0.2),
                        name='conv2')(y)
      return a, b

    def body(a, b, labels):
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a = math_ops.reduce_mean(a, axis=[2, 3]) + math_ops.reduce_mean(
          b, axis=[2, 3])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=a,
                                                          labels=labels))
      train_op = gradient_descent.GradientDescentOptimizer(0.001).minimize(
          loss)
      return a, train_op

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        labels = array_ops.placeholder(np.int32, [2])
      return a, b, labels

    init_values = [
        np.ones([2, 32, 32, 4]),
        np.ones([2, 32, 32, 4]),
        np.ones([2])
    ]

    conv_classifications = [0, 2, 1, 2]
    compute_sets = ['/multi-conv*/MultiConv']
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        partial_compute_sets=compute_sets)

  @test_util.deprecated_graph_mode_only
  def testTrainingDepthwiseTraining(self):
    np.random.seed(1234)
    w1 = np.random.random_sample([4, 4, 4, 2])
    w2 = np.random.random_sample([4, 4, 4, 2])

    def convs(x, y):
      a = layers.DepthwiseConv2D(
          8,
          2,
          depth_multiplier=2,
          use_bias=True,
          depthwise_initializer=init_ops.constant_initializer(w1),
          bias_initializer=init_ops.constant_initializer(0.5),
          name='conv1')(x)
      b = layers.DepthwiseConv2D(
          4,
          2,
          depth_multiplier=2,
          use_bias=True,
          depthwise_initializer=init_ops.constant_initializer(w2),
          bias_initializer=init_ops.constant_initializer(0.2),
          name='conv2')(y)
      return a, b

    def body(a, b, labels):
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a = math_ops.reduce_mean(a, axis=[1, 2]) + math_ops.reduce_mean(
          b, axis=[1, 2])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=a,
                                                          labels=labels))
      train_op = gradient_descent.GradientDescentOptimizer(0.001).minimize(
          loss)
      return a, train_op

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        labels = array_ops.placeholder(np.int32, [2])
      return a, b, labels

    init_values = [
        np.ones([2, 32, 32, 4]),
        np.ones([2, 32, 32, 4]),
        np.ones([2])
    ]
    # TODO(T22014): the bwd conv is not classified correctly.
    conv_classifications = [1, 2, 0, 2]
    compute_sets = ['/multi-conv*/MultiConv']
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        partial_compute_sets=compute_sets)

  @test_util.deprecated_graph_mode_only
  def testMixedConvs(self):
    np.random.seed(1234)
    h_w1 = np.random.random_sample([1, 1, 4, 2])
    h_w2 = np.random.random_sample([1, 1, 1, 4])

    def convs(a, b, w1, w2):
      a = nn.conv2d(a, w1, 1, padding='VALID')
      b = nn.conv2d_transpose(b, w2, [2, 32, 32, 4], 1)
      return a, b

    def body(a, b):
      w1 = variable_scope.get_variable(
          "w1",
          dtype=np.float32,
          shape=[1, 1, 4, 2],
          initializer=init_ops.constant_initializer(h_w1))
      w2 = variable_scope.get_variable(
          "w2",
          dtype=np.float32,
          shape=[1, 1, 4, 2],
          initializer=init_ops.constant_initializer(h_w2))
      a, b = _test_multi_conv_wrapper(convs)(a, b, w1, w2)

      a = math_ops.reduce_mean(a) + math_ops.reduce_mean(b)
      return a, control_flow_ops.no_op()

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 2])
      return a, b

    init_values = [np.ones([2, 32, 32, 4]), np.ones([2, 32, 32, 2])]

    conv_classifications = [1, 0, 0, 0]
    compute_sets = ['/multi-conv*/MultiConv']
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        partial_compute_sets=compute_sets)

  @test_util.deprecated_graph_mode_only
  def testReuse(self):
    def convs(x, y):
      a = layers.Conv2D(2,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(x)
      b = layers.Conv2D(4,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv2')(y)
      return a, b

    def body(a, b):
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a = array_ops.concat([a, a], axis=-1)
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a = math_ops.reduce_mean(a, axis=[2, 3]) + math_ops.reduce_mean(
          b, axis=[2, 3])
      return a, control_flow_ops.no_op()

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 4])
      return a, b

    init_values = [np.ones([2, 32, 32, 4]), np.ones([2, 32, 32, 4])]

    conv_classifications = [2, 0, 0, 0]
    # Note how there is only one multiconv instruction.
    compute_sets = [
        '/multi-conv/MultiConv_',
        'ipu/Mean*/fusion*/Reduce',
        'ipu/add',
    ]
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        compute_sets=compute_sets)

  @test_util.deprecated_graph_mode_only
  def testDifferentDataTypes(self):
    def convs(x, y):
      a = layers.Conv2D(2,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(x)
      a = math_ops.cast(a, dtype=np.float32)
      b = layers.Conv2D(2,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv2')(y)
      return a, b

    def body(a, b):
      a, b = _test_multi_conv_wrapper(convs)(a, b)
      a = math_ops.reduce_mean(a, axis=[2, 3]) + math_ops.reduce_mean(
          b, axis=[2, 3])
      return a, control_flow_ops.no_op()

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float16, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 4])
      return a, b

    init_values = [np.ones([2, 32, 32, 4]), np.ones([2, 32, 32, 4])]

    conv_classifications = [1, 0, 0, 0]
    compute_sets = ['/multi-conv*/MultiConv']
    _compare_ipu_to_cpu(self,
                        body,
                        inputs_fn,
                        init_values,
                        conv_classifications,
                        partial_compute_sets=compute_sets)

  @test_util.deprecated_graph_mode_only
  def testDataDependencies(self):
    def convs(x):
      a = layers.Conv2D(2,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(x)
      b = layers.Conv2D(4,
                        1,
                        use_bias=False,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv2')(a)
      return b

    def body(a):
      a = _test_multi_conv_wrapper(convs)(a)
      return a, control_flow_ops.no_op()

    def inputs_fn():
      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
      return [a]

    init_values = [np.ones([2, 32, 32, 4])]

    conv_classifications = [1, 0, 0, 0]
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "A MultiConvolution operation requires all convolutions to be "
        "independent of each other"):
      _compare_ipu_to_cpu(self, body, inputs_fn, init_values,
                          conv_classifications)

  @test_util.deprecated_graph_mode_only
  def testOptions(self):
    with self.test_session() as session:
      np.random.seed(1234)
      h_w1 = np.random.random_sample([1, 1, 4, 2])
      h_w2 = np.random.random_sample([1, 1, 1, 4])

      @ipu.nn_ops.multi_conv(options={"invalidFlag": "yes"})
      def convs(a, b, w1, w2):
        a = nn.conv2d(a, w1, 1, padding='VALID')
        b = nn.conv2d_transpose(b, w2, [2, 32, 32, 4], 1)
        return a, b

      def body(a, b):
        cfg = ipu.config.IPUConfig()
        cfg.ipu_model.compile_ipu_code = False
        cfg.configure_ipu_system()

        w1 = variable_scope.get_variable(
            "w1",
            dtype=np.float32,
            shape=[1, 1, 4, 2],
            initializer=init_ops.constant_initializer(h_w1))
        w2 = variable_scope.get_variable(
            "w2",
            dtype=np.float32,
            shape=[1, 1, 4, 2],
            initializer=init_ops.constant_initializer(h_w2))
        a, b = convs(a, b, w1, w2)
        option_flags = a.op.get_attr("option_flags")
        option_flags_proto = json_format.Parse(
            option_flags, option_flag_pb2.PoplarOptionFlags())
        self.assertEqual(len(option_flags_proto.flags), 1)
        self.assertEqual(option_flags_proto.flags[0].option, "invalidFlag")
        self.assertEqual(option_flags_proto.flags[0].value, "yes")
        return a, b

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [2, 32, 32, 4])
        b = array_ops.placeholder(np.float32, [2, 32, 32, 2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b])

      tu.move_variable_initialization_to_cpu()
      session.run(variables.global_variables_initializer())
      with self.assertRaisesRegex(
          Exception,
          r"\[Poplar\]\[Build graph\] invalid_option: Unrecognised option "
          r"\'invalidFlag\'"):
        session.run(res, {x: np.ones(x.shape) for x in [a, b]})


if __name__ == "__main__":
  googletest.main()

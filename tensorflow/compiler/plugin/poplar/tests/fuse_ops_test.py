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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pva
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu.config import IPUConfig, SchedulingAlgorithm
from tensorflow.python.layers import normalization as layers_norm
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import gradient_descent
from tensorflow.python.compiler.xla import xla
from tensorflow.random import normal


class IpuFuseOpsTest(xla_test.XLATestCase):
  def testSigmoid(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = math_ops.sigmoid(pa)

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

    report = pva.openReport(report_helper.find_report())
    ok = ['Sigmoid/sigmoid/Nonlinearity']
    self.assert_all_compute_sets_and_list(report, ok)

  def testSigmoidNotInplace(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = math_ops.sigmoid(pa) + pa

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [-5.997527, 0.5, 6.997527])

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = ['Sigmoid/sigmoid/Nonlinearity', 'add/add.*/Add']
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testSigmoidGrad(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="grad")
        pb = array_ops.placeholder(np.float32, [3], name="in")
        c = gen_math_ops.sigmoid_grad(pa, pb)

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.25, 0.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['SigmoidGrad/sigmoid-grad/NonLinearityGrad']
    self.assert_all_compute_sets_and_list(report, ok)

  def testRelu(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = nn_ops.relu(pa)

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['Relu/relu/Nonlinearity']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReluExpr(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def relu_like(x):
      return math_ops.maximum(x, array_ops.zeros_like(x))

    with self.session() as sess:
      input_values = np.ones((1, 4, 4, 2))

      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        lr = array_ops.placeholder(np.float32, shape=[])
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            kernel_initializer=init_ops.ones_initializer(),
                            name="a")(x)
          y = relu_like(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      fe = {
          x: input_values,
          lr: 0.1,
      }
      sess.run((loss, train), fe)

    report = pva.openReport(report_helper.find_report())
    ok = [
        'GradientDescent/update_vs',
        'Sum/reduce',
        'Maximum/relu/Nonlinearity',
        'gradients/vs/Maximum_grad',
        'gradients/vs/a/Conv2D_grad',
        'vs/a/BiasAdd',
        'vs/a/Conv2D',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testReluNotInPlace(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = nn_ops.relu(pa) + pa

      fd = {pa: [1, -2, 1]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2, -2, 2])

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = ['Relu/relu/Nonlinearity', 'add/add.*/Add']
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testReluNotInPlace2(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [5], name="a")
        b = array_ops.concat([pa, pa], axis=0)
        c = nn_ops.relu(b)

      fd = {pa: [-2, -1, 0, 1, 2]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0, 0, 0, 1, 2, 0, 0, 0, 1, 2])
      self.assertTrue(len(result) == 10)

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = ['Relu/relu/Nonlinearity']
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testReluGrad(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="grad")
        pb = array_ops.placeholder(np.float32, [3], name="in")
        c = gen_nn_ops.relu_grad(pa, pb)

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['ReluGrad/relu-grad/NonLinearityGrad']
    self.assert_all_compute_sets_and_list(report, ok)

  def testMaxPool(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        c = nn.max_pool(pa,
                        ksize=[1, 1, 5, 5],
                        strides=[1, 1, 2, 2],
                        data_format='NCHW',
                        padding='SAME',
                        name="max")

      fd = {
          pa: np.ones([1, 1, 10, 10]),
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.ones([1, 1, 5, 5]))

    report = pva.openReport(report_helper.find_report())
    ok = ['max/max-pool*/maxPool5x5']
    self.assert_all_compute_sets_and_list(report, ok)

  def testFwdAndBwdMaxPool(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      input_values = np.arange(16).reshape(1, 4, 4, 1)
      output_grad = np.full((1, 2, 2, 1), 0.1)

      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 4, 4, 1], name="a")
        pb = array_ops.placeholder(np.float32, [1, 2, 2, 1], name="b")
        c = nn.max_pool(pa,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        data_format='NCHW',
                        padding='SAME')
        d = gen_nn_ops.max_pool_grad(pa,
                                     c,
                                     pb,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     data_format='NCHW',
                                     padding='SAME')

      fe = {
          pa: input_values,
          pb: output_grad,
      }
      output, input_grad = sess.run((c, d), fe)
      self.assertAllClose(output, [[[[5.], [7.]], [[13.], [15.]]]])
      self.assertAllClose(
          input_grad, [[[[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]],
                        [[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]]]])

    report = pva.openReport(report_helper.find_report())
    ok = [
        'MaxPool/max-pool*/maxPool2x2/',
        'MaxPoolGrad/max-pool-grad*/maxPool2x2'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledAddTo(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        const = array_ops.constant(2.0, np.float16)
        c = pa + pb * const

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledSubtractFrom(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        const = array_ops.constant(2.0, np.float16)
        # note how const operand index varies compared to testScaledAddTo
        # still should match as it will be reordered
        c = pa - const * pb

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['sub/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledAddToVariable(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        pc = array_ops.placeholder(np.float16, [1])
        c = pa + pb * pc

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledSubtractFromVariable(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        pc = array_ops.placeholder(np.float16, [1])
        c = pa - pc * pb

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['sub/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testConvolutionBiasApply(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion.*/Reduce'
    ]
    # pylint: enable=line-too-long
    self.assert_compute_sets_contain_list(report, ok)

  def testConvolutionBiasApplyVariableLR(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.scheduling.algorithm = SchedulingAlgorithm.POST_ORDER
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        lr = array_ops.placeholder(np.float32, shape=[])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2]), lr: 0.1})

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'vs/conv2d/BiasAdd/fusion*/Op/Add',
        'vs/conv2d_1/BiasAdd/fusion.1/Op/Add',
        'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion*/Reduce',
        'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion*/negate/Op/Negate',
        'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/multiply',
        'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/fusion*/Subtract',
        'vs/conv2d/BiasAdd/fusion*/Op/Add',
        'Sum/reduce*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
        'Sum/reduce*/ReduceFinalStage/IntermediateToOutput/Reduce',
        'gradients/vs/conv2d/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4',
        'gradients/vs/conv2d/Conv2D_grad/Conv2DBackpropFilter/fusion*/AddTo',
        'vs/conv2d/Conv2D/convolution*/Conv_1x1',
        'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropInput/weights-transpose-chans-flip-x-y*/WeightsTransposeChansFlipXY/WeightsTranspose'
    ]
    # pylint: enable=line-too-long

    self.assert_all_compute_sets_and_list(report, ok)

  def testAvgPoolValid(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      np.random.seed(0)
      shape = [1, 10, 10, 1]
      data = np.random.uniform(0, 1, shape)
      # The expected answer was generated using TF on the cpu
      expected = [[[[0.47279388]]]]

      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, shape, name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 10, 10, 1],
                             strides=[1, 1, 1, 1],
                             data_format='NHWC',
                             padding='VALID',
                             name="avg")

      sess.run(variables.global_variables_initializer())

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

    report = pva.openReport(report_helper.find_report())
    ok = ['avg/avg-pool*/avgPool10x10']
    self.assert_all_compute_sets_and_list(report, ok)

  def testAvgPoolValidWithBroadcast(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      np.random.seed(0)
      shape = [1, 10, 10, 1]
      data = np.random.uniform(0, 1, shape)
      # The expected answer was generated using TF on the cpu
      expected = [[[[0.52647954], [0.44196457], [0.49284577]],
                   [[0.44039682], [0.44067329], [0.44934618]],
                   [[0.46444583], [0.45419583], [0.38236427]]]]

      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, shape, name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 5, 5, 1],
                             strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='VALID',
                             name="avg")

      sess.run(variables.global_variables_initializer())

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

    report = pva.openReport(report_helper.find_report())
    ok = ['avg/avg-pool*/avgPool5x5']
    self.assert_all_compute_sets_and_list(report, ok)

  def testAvgPoolSameWithReshape(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      np.random.seed(0)
      shape = [1, 10, 10, 1]
      data = np.random.uniform(0, 1, shape)
      # The expected answer was generated using TF on the cpu
      expected = [[[[0.64431685], [0.51738459], [0.49705142], [0.60235918],
                    [0.73694557]],
                   [[0.57755166], [0.47387227], [0.40451217], [0.4876942],
                    [0.55843753]],
                   [[0.49037799], [0.4466258], [0.35829377], [0.40070742],
                    [0.37205362]],
                   [[0.47563809], [0.4075647], [0.34894851], [0.35470542],
                    [0.3322109]],
                   [[0.52914065], [0.45464769], [0.38156652], [0.32455513],
                    [0.33199897]]]]

      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, shape, name="a")
        output = nn.avg_pool(pa,
                             ksize=[1, 5, 5, 1],
                             strides=[1, 2, 2, 1],
                             data_format='NHWC',
                             padding='SAME',
                             name="avg")

      sess.run(variables.global_variables_initializer())

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

    report = pva.openReport(report_helper.find_report())
    ok = ['avg/avg-pool*/avgPool5x5']
    self.assert_all_compute_sets_and_list(report, ok)

  def testFullyConnectedWithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[2, 2])
        weights = array_ops.placeholder(np.float32, shape=[2, 2])
        bias = array_ops.placeholder(np.float32, shape=[2])
        x_new = nn.xw_plus_b(x, weights, bias)

      out = sess.run(x_new, {
          x: np.full([2, 2], 3),
          weights: np.full([2, 2], 4),
          bias: np.ones([2]),
      })
      self.assertAllClose(np.full([2, 2], 25), out)

    report = pva.openReport(report_helper.find_report())
    ok = ['xw_plus_b/MatMul/dot/Conv_1', 'xw_plus_b/fusion/Op/Add']
    self.assert_all_compute_sets_and_list(report, ok)

  def testBatchedMatmulWithBias(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[2, 2, 2])
        weights = array_ops.placeholder(np.float32, shape=[2, 2])
        bias = array_ops.placeholder(np.float32, shape=[2])
        x_new = x @ weights + bias

      out = sess.run(
          x_new, {
              x: np.full([2, 2, 2], 3),
              weights: np.full([2, 2], 4),
              bias: np.ones([2]),
          })
      self.assertAllClose(np.full([2, 2, 2], 25), out)

    report = pva.openReport(report_helper.find_report())
    ok = ['matmul/dot*/Conv', 'add/fusion/Op/Add']
    self.assert_all_compute_sets_and_list(report, ok)

  def testConvWithBnAndRelu(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = layers_norm.batch_normalization(y, fused=True)
          y = nn_ops.relu(y)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

    report = pva.openReport(report_helper.find_report())
    ok = [
        'vs/conv2d/Conv2D/convolution/Conv_1x1', 'vs/conv2d/BiasAdd',
        'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference/',
        'vs/Relu/relu/Nonlinearity'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testBiasApplyFixedLR(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      input_values = np.ones((1, 4, 4, 2))

      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer(),
                            bias_initializer=init_ops.ones_initializer(),
                            name="a")(x)
          y = nn.relu(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      fe = {
          x: input_values,
      }
      sess.run((loss, train), fe)
      tvars = variables.global_variables()
      tvars_vals = sess.run(tvars)

      found = False
      for var, val in zip(tvars, tvars_vals):
        if var.name == "vs/a/bias:0":
          # Value computed using the CPU backend
          self.assertAllClose(val, [-0.6, -0.6])
          found = True
      self.assertTrue(found)

  def testBiasApplyVariableLR(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      input_values = np.ones((1, 4, 4, 2))

      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float16, shape=[1, 4, 4, 2])
        lr = array_ops.placeholder(np.float16, shape=[])
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer(),
                            bias_initializer=init_ops.ones_initializer(),
                            name="a")(x)
          y = nn.relu(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      fe = {
          x: input_values,
          lr: 0.1,
      }
      sess.run((loss, train), fe)
      tvars = variables.global_variables()
      tvars_vals = sess.run(tvars)

      found = False
      for var, val in zip(tvars, tvars_vals):
        if var.name == "vs/a/bias:0":
          # Value computed using the CPU backend
          self.assertAllClose(val, [-0.6, -0.6], atol=0.001)
          found = True
      self.assertTrue(found)

  def testUnsortedSegmentSumConstLR(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def network(x, y1, y2):
        del x
        with variable_scope.variable_scope("vs", use_resource=True):
          w1 = variable_scope.get_variable(
              "w1",
              shape=[10, 200],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(1))
          g1 = array_ops.gather(w1, y1)
          g2 = array_ops.gather(w1, y2)

          a = math_ops.reduce_sum(g1 + g2)

        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        grads = [a]
        grads = [
            gradients_impl.gradients(g, variables.trainable_variables())[0]
            for g in grads
        ]
        grads = [array_ops.expand_dims(g, 0) for g in grads]
        grad = array_ops.concat(grads, axis=0)
        grad = math_ops.reduce_mean(grad, 0)
        train = optimizer.apply_gradients([(grad, w1)])
        return a, train

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[10, 200])
        y1 = array_ops.placeholder(np.int32, shape=[10])
        y2 = array_ops.placeholder(np.int32, shape=[10])

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      out = sess.run(r, {
          x: np.ones(x.shape),
          y1: np.ones(y1.shape),
          y2: np.ones(y2.shape),
      })
      self.assertAllClose(out, [-4000.0])

    report = pva.openReport(report_helper.find_report())
    ok = [
        'ExpandDims/input/multi-update-add*/multiUpdateAdd',
        'vs/Gather*/multi-slice',
        'vs/add/add*/Add',
        'vs/Sum/reduce*/Reduce',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testUnsortedSegmentSumVariableLR(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def network(x, y1, y2, lr):
        del x
        with variable_scope.variable_scope("vs", use_resource=True):
          w1 = variable_scope.get_variable(
              "w1",
              shape=[10, 200],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(1))
          g1 = array_ops.gather(w1, y1)
          g2 = array_ops.gather(w1, y2)

          a = math_ops.reduce_sum(g1 + g2)

        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        grads = [a]
        grads = [
            gradients_impl.gradients(g, variables.trainable_variables())[0]
            for g in grads
        ]
        grads = [array_ops.expand_dims(g, 0) for g in grads]
        grad = array_ops.concat(grads, axis=0)
        grad = math_ops.reduce_mean(grad, 0)
        train = optimizer.apply_gradients([(grad, w1)])
        return a, train

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[10, 200])
        y1 = array_ops.placeholder(np.int32, shape=[10])
        y2 = array_ops.placeholder(np.int32, shape=[10])
        lr = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2, lr])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y1: np.ones(y1.shape),
              y2: np.ones(y2.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [-4000.0])

    report = pva.openReport(report_helper.find_report())
    ok = [
        '/negate/Op/Negate',
        'ExpandDims/input/multi-update-add*/multiUpdateAdd',
        'vs/Gather*/multi-slice',
        'vs/add/add*/Add',
        'vs/Sum/reduce*/Reduce',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testScatterSingleLookup(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def network(x, y, la, lr):
        del x
        with variable_scope.variable_scope("vs", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[10, 200],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(2.))
          g = nn.embedding_lookup(w, y)

          ce = losses.absolute_difference(labels=la, predictions=g)
          loss = math_ops.reduce_mean(ce)

        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        return loss, train

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[1, 100, 100, 2])
        y = array_ops.placeholder(np.int32, shape=[10])
        la = array_ops.placeholder(np.float32, shape=[10, 200])
        lr = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y, la, lr])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y: np.ones(y.shape),
              la: np.ones(la.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [1.0])

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'GradientDescent/update_vs/w/Neg/negate*/Op/Negate',
        'GradientDescent/update_vs/w/mul/fusion*/Op/Multiply',
        'GradientDescent/update_vs/w/ResourceScatterAdd/multi-update-add*/multiUpdateAdd',
        'gradients/vs/absolute_difference/Abs_grad/Sign',
        'gradients/vs/absolute_difference/Abs_grad/mul/fusion',
        'vs/embedding_lookup/multi-slice',
        'vs/absolute_difference/Sub/subtract.*/Subtract',
        'vs/absolute_difference/Abs/abs.*/Op/Absolute',
        'vs/absolute_difference/Sum/reduce',
        'vs/absolute_difference/value/multiply',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testScatterMultipleLookupsWithReshape(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def network(x, y1, y2, la, lr):
        del x
        with variable_scope.variable_scope("vs", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[200, 10],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(2.))
          y = array_ops.reshape(w, [10, 200])
          g1 = nn.embedding_lookup(y, y1)
          g2 = nn.embedding_lookup(y, y2)
          g = array_ops.concat([g1, g2], axis=1)

          ce = losses.absolute_difference(labels=la, predictions=g)
          loss = math_ops.reduce_mean(ce)

        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        return loss, train

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[1, 100, 100, 2])
        y1 = array_ops.placeholder(np.int32, shape=[10])
        y2 = array_ops.placeholder(np.int32, shape=[10])
        la = array_ops.placeholder(np.float32, shape=[10, 400])
        lr = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2, la, lr])

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y1: np.ones(y1.shape),
              y2: np.ones(y2.shape),
              la: np.ones(la.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [1.0])

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'gradients/vs/absolute_difference/Abs_grad/Sign',
        'gradients/vs/absolute_difference/Abs_grad/mul/fusion',
        '/negate/Op/Negate',
        'gradients/vs/Reshape_grad/UnsortedSegmentSum/multi-update-add*/multiUpdateAdd',
        'vs/embedding_lookup*/multi-slice',
        'vs/absolute_difference/Sub/subtract.*/Subtract',
        'vs/absolute_difference/Abs/abs.*/Op/Absolute',
        'vs/absolute_difference/Sum/reduce',
        'vs/absolute_difference/value/multiply',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledAddaXbY(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        const_a = array_ops.constant(2.0, np.float16)
        const_b = array_ops.constant(3.0, np.float16)
        axby = const_a * px + const_b * py

      fd = {px: [2.0, 0.5, 1.0], py: [1.0, 2.0, 3.0]}
      result = sess.run(axby, fd)
      self.assertAllClose(result, [7.0, 7.0, 11.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledSubtractaXbY(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        const_a = array_ops.constant(2.0, np.float16)
        const_b = array_ops.constant(3.0, np.float16)
        axby = const_a * px - const_b * py

      fd = {px: [2.0, 0.5, 1.0], py: [1.0, 2.0, 3.0]}
      result = sess.run(axby, fd)
      self.assertAllClose(result, [1.0, -5.0, -7.0])

  def testScaledAddaXbYMixed(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        scale_a = array_ops.placeholder(np.float32, [])
        scale_b = array_ops.placeholder(np.float32, [])
        scale_a_16 = math_ops.cast(scale_a, dtypes.float16)
        scale_b_16 = math_ops.cast(scale_b, dtypes.float16)
        axby = scale_a_16 * px + scale_b_16 * py

      fd = {
          px: [2.0, 0.5, 1.0],
          py: [1.0, 2.0, 3.0],
          scale_a: 2.0,
          scale_b: 3.0
      }
      result = sess.run(axby, fd)
      self.assertAllClose(result, [7.0, 7.0, 11.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  # Using a scale smaller than a float16 can represent
  # to ensure the cast is not actually being performed
  def testScaledAddXbYMixedSmallScale(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        scale_b = array_ops.placeholder(np.float32, [])
        scale_b_16 = math_ops.cast(scale_b, dtypes.float16)
        xby = px + scale_b_16 * py

      # Note that the output xby is still a float16 so py is scaled up
      # to ensure the output is representable as a float16
      fd = {
          px: [4.0e-4, 1.0e-4, 2.0e-4],
          py: [1.0e4, 2.0e4, 3.0e4],
          scale_b: 3e-8
      }
      result = sess.run(xby, fd)
      self.assertAllClose(result, [7.0e-4, 7.0e-4, 11.0e-4])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledAddToVariableFor2Scales(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pa_scale = array_ops.placeholder(np.float16, [1])
        pb = array_ops.placeholder(np.float16, [3])
        pb_scale = array_ops.placeholder(np.float16, [1])
        c = pa_scale * pa + pb_scale * pb

      fd = {
          pa: [2.0, 0.5, 1.0],
          pb: [1.0, 2.0, 3.0],
          pa_scale: [2.0],
          pb_scale: [3.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [7.0, 7.0, 11.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['add/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testScaledSubtractFromVariableFor2Scales(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pa_scale = array_ops.placeholder(np.float16, [1])
        pb = array_ops.placeholder(np.float16, [3])
        pb_scale = array_ops.placeholder(np.float16, [1])
        c = pa_scale * pa - pb_scale * pb
      fd = {
          pa: [2.0, 0.5, 1.0],
          pb: [1.0, 2.0, 3.0],
          pa_scale: [2.0],
          pb_scale: [3.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [1.0, -5.0, -7.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['sub/scaled-inplace*/AddTo']
    self.assert_all_compute_sets_and_list(report, ok)

  def testPopOpNormScaleAddLiteralScalars(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def build_graph():
      k_constant6 = array_ops.constant([[0.0, 0.0], [0.0, 0.0]])
      k_constant5 = array_ops.constant([[1.0, 1.0], [1.0, 1.0]])
      k_rng4 = normal([2, 2],
                      mean=k_constant6,
                      stddev=k_constant5,
                      dtype=np.float32)
      k_constant3 = array_ops.constant([[2.0, 2.3], [1.0, 2.2]])
      k_multiply2 = k_rng4 * k_constant3
      k_constante1 = array_ops.constant([[0.1, 1.5], [0.9, 1.3]])
      return k_multiply2 + k_constante1

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        r = xla.compile(build_graph, inputs=[])
      sess.run(r)

    report = pva.openReport(report_helper.find_report())
    ok = [
        'random_normal/RandomStandardNormal/rng.*/normal',
        'mul/multiply.*/Op/Multiply', 'add/add*/Op/Add'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testSquareSum(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = math_ops.reduce_sum(pa * pa)

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, 72)

    report = pva.openReport(report_helper.find_report())
    ok = ['Sum/fusion*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testSquareSumHalfs(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [2], name="a")
        c = math_ops.reduce_mean(pa * pa)

      # These values are too big to for float16 when squared and summed
      # but since the squaring and summation happen in float32 we should
      # get back a valid f16 value (if the fusion has worked).
      fd = {pa: [2**8, 2]}
      result = sess.run(c, fd)
      self.assertAllClose(result, np.float16(32770.0))

    report = pva.openReport(report_helper.find_report())
    ok = ['Mean/fusion*/Reduce', 'Mean/multiply', 'Mean/convert*/Cast']
    self.assert_all_compute_sets_and_list(report, ok)

  def testConvolutionWithReverseWeights(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, [1, 4, 4, 2], name="x")
        lr = array_ops.placeholder(np.float32, [], name="lr")

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            2,
                            kernel_initializer=init_ops.constant_initializer(
                                [[[[1, -1], [1, 1]], [[2, -1], [-1, -2]]],
                                 [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]]),
                            use_bias=False)(x)
          y = layers.Conv2D(2,
                            1,
                            kernel_initializer=init_ops.constant_initializer([
                                [
                                    [[0.3, 0.6], [-1, 0.5]],
                                ],
                            ]),
                            use_bias=False)(y)

        loss = math_ops.reduce_mean(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      train, loss = sess.run(
          [train, loss], {
              x:
              np.array([[
                  [[1, 2], [3, 4], [-1, -2], [-3, -4]],
                  [[2, 1], [2, 1], [-1, -2], [-1, -2]],
                  [[1, 1], [2, 2], [3, 3], [4, 4]],
                  [[1, 2], [-1, -2], [-1, -1], [-2, -2]],
              ]]),
              lr:
              1
          })

      k1, k2 = sess.run(variables.trainable_variables())
      self.assertAllClose(loss, 2.3611107)
      self.assertAllClose(
          k1,
          np.array([[[[0.39999998, -0.6666667], [0.5, 1.2777778]],
                     [[1.6, -0.7777778], [-1.2, -1.8888888]]],
                    [[[0.6, 2.2222223], [2.75, 4.138889]],
                     [[0.75, 2.1388888], [2.95, 4.0277777]]]]))
      self.assertAllClose(
          k2, np.array([[[[-3.3111112, -3.0111113], [-2.7777781,
                                                     -1.277778]]]]))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'vs/conv2d_1/Conv2D/convolution*/Conv_1x1',
        'vs/conv2d/Conv2D/convolution*/Conv_2x2',
        'gradients/vs/conv2d/Conv2D_grad/Conv2DBackpropFilter/fusion',
        'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_3x3/',
        'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropFilter/fusion*/AddTo',
        'gradients/vs/conv2d_1/Conv2D_grad/Conv2DBackpropInput/weights-transpose-chans-flip-x-y.*/WeightsTransposeChansFlipXY/WeightsTranspose',
        'Mean/',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testUniformScaleAdd(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def fn():
        random_number = random_ops.random_uniform([], maxval=100)
        random_number_plus_10 = (random_number + 10.)
        output = [random_number, random_number_plus_10]
        return output

      with ops.device("/device:IPU:0"):
        op = fn()

      res = sess.run(op)
      self.assertAllClose(res[1] - res[0], 10)

  def testNormalScaleAdd(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def fn():
        random_number = random_ops.random_normal([])
        random_number_plus_10 = (random_number + 10.)
        output = [random_number, random_number_plus_10]
        return output

      with ops.device("/device:IPU:0"):
        op = fn()

      res = sess.run(op)
      self.assertAllClose(res[1] - res[0], 10)

  def initConfigure(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()
    return report_helper

  def testReductionAddScaled(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add(to_reduce, scale):
        return math_ops.multiply(
            math_ops.reduce_sum(to_reduce, axis=[0, 1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        scale = array_ops.placeholder(np.float32, None, name="scale")
        output = scaled_reduce_add(to_reduce, scale)
        fd = {to_reduce: np.ones([2, 2, 2]), scale: np.array([5.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionAddScaledFp16(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_fp16(to_reduce, scale):
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.cast(to_reduce, np.float32),
                                axis=[0, 1]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float16, [2, 2], name="to_reduce")
        scale = array_ops.placeholder(np.float32, None, name="scale")
        output = scaled_reduce_add_fp16(to_reduce, scale)
      fd = {to_reduce: np.ones([2, 2]), scale: np.array([10.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionAddScaledConstant(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_constant(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(
            math_ops.reduce_sum(to_reduce, axis=[0, 1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        output = scaled_reduce_add_constant(to_reduce)
        fd = {to_reduce: np.ones([2, 2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, 80)

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionAddScaledConstantFp16(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_fp16(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.cast(to_reduce, np.float32),
                                axis=[0, 1]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float16, [2, 2], name="to_reduce")
        output = scaled_reduce_add_fp16(to_reduce)
        fd = {to_reduce: np.ones([2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, 40)

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionAddDim12Scaled(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_sum(to_reduce, scale):
        return math_ops.multiply(math_ops.reduce_sum(to_reduce, axis=[1, 2]),
                                 scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        scale = array_ops.placeholder(np.float32, [1], name="scale")
        output = scaled_reduce_sum(to_reduce, scale)
        fd = {to_reduce: np.ones([2, 2, 2]), scale: np.array([10.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40., 40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionAddDim12ScaledConstant(self):

    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_sum(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(math_ops.reduce_sum(to_reduce, axis=[1, 2]),
                                 scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        output = scaled_reduce_sum(to_reduce)
        fd = {to_reduce: np.ones([2, 2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40., 40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaled(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add(to_reduce, scale):
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.multiply(to_reduce, to_reduce),
                                axis=[0, 1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        scale = array_ops.placeholder(np.float32, None, name="scale")
        output = scaled_reduce_add(to_reduce, scale)
        fd = {to_reduce: np.ones([2, 2, 2]), scale: np.array([5.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaledFp16(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_fp16(to_reduce, scale):
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.cast(
                math_ops.multiply(to_reduce, to_reduce), np.float32),
                                axis=[0, 1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float16, [2, 2, 2],
                                          name="to_reduce")
        scale = array_ops.placeholder(np.float32, None, name="scale")
        output = scaled_reduce_add_fp16(to_reduce, scale)
        fd = {to_reduce: np.ones([2, 2, 2]), scale: np.array([5.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaledConstant(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_constant(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.multiply(to_reduce, to_reduce),
                                axis=[0, 1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        output = scaled_reduce_add_constant(to_reduce)
        fd = {to_reduce: np.ones([2, 2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, 80)

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaledConstantFp16(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_add_fp16(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.cast(
                math_ops.multiply(to_reduce, to_reduce), np.float32),
                                axis=[0, 1]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float16, [2, 2], name="to_reduce")
        output = scaled_reduce_add_fp16(to_reduce)
        fd = {to_reduce: np.ones([2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, 40)

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaledDim12Scaled(self):
    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_sum(to_reduce, scale):
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.multiply(to_reduce, to_reduce),
                                axis=[1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        scale = array_ops.placeholder(np.float32, [1], name="scale")
        output = scaled_reduce_sum(to_reduce, scale)
        fd = {to_reduce: np.ones([2, 2, 2]), scale: np.array([10.])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40., 40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReductionSquaredAddScaledDim12ScaledConstant(self):

    report_helper = self.initConfigure()

    with self.session() as sess:

      def scaled_reduce_sum(to_reduce):
        scale = array_ops.constant(10., dtype=np.float32)
        return math_ops.multiply(
            math_ops.reduce_sum(math_ops.multiply(to_reduce, to_reduce),
                                axis=[1, 2]), scale)

      with ops.device("/device:IPU:0"):
        to_reduce = array_ops.placeholder(np.float32, [2, 2, 2],
                                          name="to_reduce")
        output = scaled_reduce_sum(to_reduce)
        fd = {to_reduce: np.ones([2, 2, 2])}
      result = sess.run(output, fd)
      self.assertAllClose(result, np.array([40., 40.]))

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'Sum/fusion/ReduceOnTile/*/Reduce']
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

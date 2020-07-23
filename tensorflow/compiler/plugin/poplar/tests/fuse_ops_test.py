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
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
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
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import gradient_descent
from tensorflow.python.compiler.xla import xla
from tensorflow.random import normal


class IpuFuseOpsTest(xla_test.XLATestCase):
  def testSigmoid(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = math_ops.sigmoid(pa)

      report = tu.ReportJSON(self, sess)

      report.reset()

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.002473, 0.5, 0.997527])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'Sigmoid/custom-call/Nonlinearity']
      report.assert_all_compute_sets_and_list(ok)

  def testSigmoidNotInplace(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = math_ops.sigmoid(pa) + pa

      report = tu.ReportJSON(self, sess)

      report.reset()

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [-5.997527, 0.5, 6.997527])

      report.parse_log(assert_len=4)

      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_XLA_Args*/arg0.*_to_Sigmoid/custom-call/Nonlinearity/out/OnTileCopy-0',
          'Sigmoid/custom-call/Nonlinearity', 'add/add.*/Add'
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testSigmoidGrad(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="grad")
        pb = array_ops.placeholder(np.float32, [3], name="in")
        c = gen_math_ops.sigmoid_grad(pa, pb)

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.25, 0.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'SigmoidGrad/custom-call/NonLinearityGrad']
      report.assert_all_compute_sets_and_list(ok)

  def testRelu(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = nn_ops.relu(pa)

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [-6.0, 0.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.0, 6.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'Relu/custom-call/Nonlinearity']
      report.assert_all_compute_sets_and_list(ok)

  def testReluNotInPlace(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="a")
        c = nn_ops.relu(pa) + pa

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [1, -2, 1]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2, -2, 2])

      report.parse_log(assert_len=4)

      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_XLA_Args*/arg0.*_to_Relu/custom-call/Nonlinearity/out/OnTileCopy-0',
          'Relu/custom-call/Nonlinearity', 'add/add.*/Add'
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testReluNotInPlace2(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [5], name="a")
        b = array_ops.concat([pa, pa], axis=0)
        c = nn_ops.relu(b)

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [-2, -1, 0, 1, 2]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0, 0, 0, 1, 2, 0, 0, 0, 1, 2])
      self.assertTrue(len(result) == 10)

      report.parse_log()

      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_XLA_Args*/arg0.*_to_Relu/custom-call/Nonlinearity/out/OnTileCopy-0',
          'Relu/custom-call/Nonlinearity'
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testReluGrad(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3], name="grad")
        pb = array_ops.placeholder(np.float32, [3], name="in")
        c = gen_nn_ops.relu_grad(pa, pb)

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [-1.0, 1.0, 6.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, 0.5, 1.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'ReluGrad/custom-call/NonLinearityGrad']
      report.assert_all_compute_sets_and_list(ok)

  def testMaxPool(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [1, 1, 10, 10], name="a")
        c = nn.max_pool(pa,
                        ksize=[1, 1, 5, 5],
                        strides=[1, 1, 2, 2],
                        data_format='NCHW',
                        padding='SAME',
                        name="max")

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: np.ones([1, 1, 10, 10]),
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.ones([1, 1, 5, 5]))

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'max/custom-call*/maxPool5x5']
      report.assert_all_compute_sets_and_list(ok)

  def testFwdAndBwdMaxPool(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()

      fe = {
          pa: input_values,
          pb: output_grad,
      }
      output, input_grad = sess.run((c, d), fe)
      self.assertAllClose(output, [[[[5.], [7.]], [[13.], [15.]]]])
      self.assertAllClose(
          input_grad, [[[[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]],
                        [[0.], [0.], [0.], [0.]], [[0.], [0.1], [0.], [0.1]]]])

      report.parse_log(assert_len=4)

      ok = [
          '__seed*', 'Copy_*', 'MaxPool/custom-call*/maxPool2x2/',
          'MaxPoolGrad/custom-call*/maxPool2x2'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testScaledAddTo(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        const = array_ops.constant(2.0, np.float16)
        c = pa + pb * const

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testScaledSubtractFrom(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        const = array_ops.constant(2.0, np.float16)
        # note how const operand index varies compared to testScaledAddTo
        # still should match as it will be reordered
        c = pa - const * pb

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'sub/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testScaledAddToVariable(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        pc = array_ops.placeholder(np.float16, [1])
        c = pa + pb * pc

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [4.0, 4.5, 7.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testScaledSubtractFromVariable(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        pc = array_ops.placeholder(np.float16, [1])
        c = pa - pc * pb

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 2.0, 3.0], pc: [2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [0.0, -3.5, -5.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'sub/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testConvolutionBiasApply(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      report.parse_log(
          assert_len=6,
          assert_msg=
          "Expected 2x compile, 1x upload, 1x load, 1x download, 1x execute")

      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion.*/Reduce'
      ]
      # pylint: enable=line-too-long
      report.assert_compute_sets_contain_list(ok)

  def testConvolutionBiasApplyVariableLR(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2]), lr: 0.1})

      report.parse_log(
          assert_len=6,
          assert_msg=
          "Expected 2x compile, 1x upload, 1x load, 1x download, 1x execute")

      # pylint: disable=line-too-long
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/BiasAdd/fusion*/Op/Add',
          'vs/conv2d_1/BiasAdd/fusion.2/Op/Add',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion.3/ReduceFinalStage/IntermediateToOutput/Reduce',
          'GradientDescent/update_vs/conv2d/bias/ResourceApplyGradientDescent/fusion*/negate/Op/Negate',
          'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/multiply*/Op/Multiply',
          'GradientDescent/update_vs/conv2d_1/bias/ResourceApplyGradientDescent/fusion*/Subtract',
          'vs/conv2d/BiasAdd/fusion*/Op/Add',
          'Sum/reduce*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv2d*/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4/Transpose',
          'gradients/vs/conv2d*/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4/Convolve',
          'gradients/vs/conv2d*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Transpose',
          'gradients/vs/conv2d*/Conv2D_grad/Conv2DBackpropFilter/fusion*/AddTo',
          'gradients/vs/conv2d*/Conv2D_grad/Conv2DBackpropInput/fusion*/*Transpose',
          'vs/conv2d/Conv2D/convolution*/Conv_1x1'
      ]
      # pylint: enable=line-too-long

      report.assert_all_compute_sets_and_list(ok)

  def testAvgPoolValid(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'avg/custom-call*/avgPool10x10']
      report.assert_all_compute_sets_and_list(ok)

  def testAvgPoolValidWithBroadcast(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'avg/custom-call*/avgPool5x5']
      report.assert_all_compute_sets_and_list(ok)

  def testAvgPoolSameWithReshape(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      fd = {pa: data}
      result = sess.run(output, fd)
      self.assertAllClose(result, expected)

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'avg/custom-call*/avgPool5x5']
      report.assert_all_compute_sets_and_list(ok)

  def testFullyConnectedWithBias(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[2, 2])
        weights = array_ops.placeholder(np.float32, shape=[2, 2])
        bias = array_ops.placeholder(np.float32, shape=[2])
        x_new = nn.xw_plus_b(x, weights, bias)

      report = tu.ReportJSON(self, sess)
      report.reset()

      out = sess.run(x_new, {
          x: np.full([2, 2], 3),
          weights: np.full([2, 2], 4),
          bias: np.ones([2]),
      })
      self.assertAllClose(np.full([2, 2], 25), out)

      report.parse_log(
          assert_len=4,
          assert_msg="Expected 1x compile, 1x load, 1x download, 1x execute")

      ok = [
          '__seed*', 'host-exchange-local-copy',
          'xw_plus_b/MatMul/dot.*/Conv_1/Convolve', 'xw_plus_b/fusion/Op/Add'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBatchedMatmulWithBias(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[2, 2, 2])
        weights = array_ops.placeholder(np.float32, shape=[2, 2])
        bias = array_ops.placeholder(np.float32, shape=[2])
        x_new = x @ weights + bias

      report = tu.ReportJSON(self, sess)
      report.reset()

      out = sess.run(
          x_new, {
              x: np.full([2, 2, 2], 3),
              weights: np.full([2, 2], 4),
              bias: np.ones([2]),
          })
      self.assertAllClose(np.full([2, 2, 2], 25), out)

      report.parse_log(
          assert_len=4,
          assert_msg="Expected 1x compile, 1x load, 1x download, 1x execute")

      ok = [
          '__seed*', 'host-exchange-local-copy', 'Copy_', 'matmul/dot*/Conv',
          'add/fusion/Op/Add'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testConvWithBnAndRelu(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log(
          assert_len=6,
          assert_msg=
          "Expected 2x compile, 1x upload, 1x load, 1x download, 1x execute")

      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d/BiasAdd',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference.*/',
          'vs/Relu/custom-call/Nonlinearity'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBiasApplyFixedLR(self):
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())
      report.reset()
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

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())
      report.reset()
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

      report = tu.ReportJSON(self, sess)

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2])

      sess.run(variables.global_variables_initializer())
      report.reset()
      out = sess.run(r, {
          x: np.ones(x.shape),
          y1: np.ones(y1.shape),
          y2: np.ones(y2.shape),
      })
      self.assertAllClose(out, [-4000.0])

      report.parse_log(assert_len=6)

      ok = [
          '__seed*',
          'host-exchange-local-copy-*/OnTileCopy-0',
          '/negate/Op/Negate',
          'ExpandDims/input/custom-call.3/multiUpdateAdd',
          'Copy_*/OnTileCopy',
          'vs/Gather*/gather.*/multiSlice',
          'vs/add/add*/Add',
          'vs/Sum/reduce*/Reduce',
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testUnsortedSegmentSumVariableLR(self):
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

      report = tu.ReportJSON(self, sess)

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2, lr])

      sess.run(variables.global_variables_initializer())
      report.reset()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y1: np.ones(y1.shape),
              y2: np.ones(y2.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [-4000.0])

      report.parse_log()

      ok = [
          '__seed*',
          'host-exchange-local-copy-*/OnTileCopy-0',
          '/negate/Op/Negate',
          'ExpandDims/input/custom-call.3/multiUpdateAdd',
          'Copy_*/OnTileCopy-0',
          'vs/Gather*/gather.*/multiSlice',
          'vs/add/add*/Add',
          'vs/Sum/reduce*/Reduce',
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testScatterSingleLookup(self):
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

      report = tu.ReportJSON(self, sess)

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y, la, lr])

      sess.run(variables.global_variables_initializer())
      report.reset()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y: np.ones(y.shape),
              la: np.ones(la.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [1.0])

      report.parse_log()

      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'GradientDescent/update_vs/w/Neg/negate*/Op/Negate',
          'GradientDescent/update_vs/w/mul/fusion*/Op/Multiply',
          'GradientDescent/update_vs/w/ResourceScatterAdd/custom-call*/multiUpdateAdd',
          'gradients/vs/absolute_difference/Abs_grad/Sign',
          'gradients/vs/absolute_difference/Abs_grad/mul/fusion',
          'vs/embedding_lookup/gather.*/multiSlice',
          'vs/absolute_difference/Sub/subtract.*/Subtract',
          'vs/absolute_difference/Abs/abs.*/Op/Absolute',
          'vs/absolute_difference/Sum/reduce',
          'vs/absolute_difference/value/multiply',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testScatterMultipleLookupsWithReshape(self):
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

      report = tu.ReportJSON(self, sess)

      with ops.device("/device:IPU:0"):
        r = xla.compile(network, inputs=[x, y1, y2, la, lr])

      sess.run(variables.global_variables_initializer())
      report.reset()
      out = sess.run(
          r, {
              x: np.ones(x.shape),
              y1: np.ones(y1.shape),
              y2: np.ones(y2.shape),
              la: np.ones(la.shape),
              lr: 0.1,
          })
      self.assertAllClose(out, [1.0])

      report.parse_log()
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'host-exchange-local-copy-*/OnTileCopy-0',
          'gradients/vs/absolute_difference/Abs_grad/Sign',
          'gradients/vs/absolute_difference/Abs_grad/mul/fusion',
          '/negate/Op/Negate',
          'gradients/vs/Reshape_grad/Reshape/tensor/custom*/multiUpdateAdd',
          'vs/embedding_lookup/gather.*/multiSlice',
          'vs/embedding_lookup_1/gather.*/multiSlice',
          'vs/absolute_difference/Sub/subtract.*/Subtract',
          'vs/absolute_difference/Abs/abs.*/Op/Absolute',
          'vs/absolute_difference/Sum/reduce',
          'vs/absolute_difference/value/multiply',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testScaledAddaXbY(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        const_a = array_ops.constant(2.0, np.float16)
        const_b = array_ops.constant(3.0, np.float16)
        axby = const_a * px + const_b * py

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {px: [2.0, 0.5, 1.0], py: [1.0, 2.0, 3.0]}
      result = sess.run(axby, fd)
      self.assertAllClose(result, [7.0, 7.0, 11.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testScaledSubtractaXbY(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        px = array_ops.placeholder(np.float16, [3])
        py = array_ops.placeholder(np.float16, [3])
        const_a = array_ops.constant(2.0, np.float16)
        const_b = array_ops.constant(3.0, np.float16)
        axby = const_a * px - const_b * py

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {px: [2.0, 0.5, 1.0], py: [1.0, 2.0, 3.0]}
      result = sess.run(axby, fd)
      self.assertAllClose(result, [1.0, -5.0, -7.0])

      report.parse_log(assert_len=4)

  def testScaledAddToVariableFor2Scales(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pa_scale = array_ops.placeholder(np.float16, [1])
        pb = array_ops.placeholder(np.float16, [3])
        pb_scale = array_ops.placeholder(np.float16, [1])
        c = pa_scale * pa + pb_scale * pb

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: [2.0, 0.5, 1.0],
          pb: [1.0, 2.0, 3.0],
          pa_scale: [2.0],
          pb_scale: [3.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [7.0, 7.0, 11.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'add/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)

  def testScaledSubtractFromVariableFor2Scales(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pa_scale = array_ops.placeholder(np.float16, [1])
        pb = array_ops.placeholder(np.float16, [3])
        pb_scale = array_ops.placeholder(np.float16, [1])
        c = pa_scale * pa - pb_scale * pb
      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: [2.0, 0.5, 1.0],
          pb: [1.0, 2.0, 3.0],
          pa_scale: [2.0],
          pb_scale: [3.0]
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, [1.0, -5.0, -7.0])

      report.parse_log(assert_len=4)

      ok = ['__seed*', 'host-exchange-local-copy-', 'sub/fusion/AddTo']
      report.assert_all_compute_sets_and_list(ok)


def testPopOpNormScaleAddLiteralScalars(self):
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
    report = tu.ReportJSON(self, sess)
    report.reset()
    sess.run(r)

    report.parse_log()
    ok = [
        'random_normal/RandomStandardNormal/rng.*/normal',
        'mul/multiply.*/Op/Multiply', 'add/add.*/Op/Add', '__seed*'
    ]
    report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

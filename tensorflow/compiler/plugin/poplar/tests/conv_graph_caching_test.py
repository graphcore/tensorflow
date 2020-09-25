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

import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


class ConvGraphCachingTest(xla_test.XLATestCase):
  def testConvolutionsMatch(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer())(y)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Would fail if there were two convolutions in the graph as they would be
      # called conv2d and conv2d_1
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'Copy_'
      ]
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [2, 0, 0, 0])

  def testConvolutionsDontMatchDifferentTypes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            dtype=np.float32)(x)
          y = math_ops.cast(y, np.float16)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            dtype=np.float16)(y)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'Copy_',
          'Copy_vs/*/OnTileCopy-0', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/Cast/convert.*/Cast', 'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [2, 0, 0, 0])

  def testConvolutionsDontMatchDifferentShapes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = array_ops.reshape(y, [1, 2, 8, 2])
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer())(y)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [2, 0, 0, 0])

  def testConvolutionsDontMatchDifferentConvParams(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            strides=(2, 1),
                            kernel_initializer=init_ops.ones_initializer())(y)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [2, 0, 0, 0])

  def testConvolutionsMatchFwdBwdWu(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv1')(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv2')(y)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv3')(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv1/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4/*',
          'gradients/vs/conv1/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Transpose',
          'gradients/vs/conv1/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropInput/fusion*/*Transpose',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [0, 3, 2, 3])

  def testConvolutionsMatchFwdBwdWuVariableLR(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        lr = array_ops.placeholder(np.float32, shape=[])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv1')(x)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv2')(y)
          y = layers.Conv2D(2,
                            1,
                            use_bias=False,
                            kernel_initializer=init_ops.ones_initializer(),
                            name='conv3')(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2]), lr: 0.1})

      report.parse_log()

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
          'vs/conv*/Conv2D/convolution*/Conv_1x1',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropInput/fusion*/*Transpose',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Transpose',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

      self.assertAllEqual(report.get_ml_type_counts(), [0, 3, 2, 3])

  def testConvolutionApply(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        filter_sizes = constant_op.constant([2, 2, 3, 5], np.int32)
        input1 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        input2 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        input3 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        grads1 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        grads2 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        grads3 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        weights1 = array_ops.placeholder(np.float32, [2, 2, 3, 5])
        weights2 = array_ops.placeholder(np.float32, [2, 2, 3, 5])
        weights3 = array_ops.placeholder(np.float32, [2, 2, 3, 5])
        vlr = array_ops.placeholder(np.float32, [])

        def conv_scaled_inplace(input_values, grads, weights, lr):
          return weights - lr * nn_ops.conv2d_backprop_filter(
              input_values,
              filter_sizes,
              grads,
              strides=[1, 1, 1, 1],
              padding="SAME")

        result = (conv_scaled_inplace(input1, grads1, weights1, vlr) +
                  conv_scaled_inplace(input2, grads2, weights2, 0.1) +
                  conv_scaled_inplace(input3, grads3, weights3, 0.2))

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      r = sess.run(
          result, {
              input1: np.ones([2, 8, 8, 3]),
              input2: np.ones([2, 8, 8, 3]),
              input3: np.ones([2, 8, 8, 3]),
              grads1: np.ones([2, 8, 8, 5]),
              grads2: np.ones([2, 8, 8, 5]),
              grads3: np.ones([2, 8, 8, 5]),
              weights1: np.ones([2, 2, 3, 5]),
              weights2: np.ones([2, 2, 3, 5]),
              weights3: np.ones([2, 2, 3, 5]),
              vlr: 0.1,
          })
      # yapf: disable
      self.assertAllClose(r, [[[[-48.2, -48.2, -48.2, -48.2, -48.2],
                                [-48.2, -48.2, -48.2, -48.2, -48.2],
                                [-48.2, -48.2, -48.2, -48.2, -48.2],],
                               [[-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],],],
                              [[[-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],],
                               [[-36.2, -36.2, -36.2, -36.2, -36.2],
                                [-36.2, -36.2, -36.2, -36.2, -36.2],
                                [-36.2, -36.2, -36.2, -36.2, -36.2],]]])
      # yapf: enable
      report.parse_log()

      report.assert_compute_sets_matches('*Convolve', 1)

  def testConvolutionEvenWhenNotInplace(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        filter_sizes = constant_op.constant([2, 2, 3, 5], np.int32)
        input1 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        input2 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        input3 = array_ops.placeholder(np.float32, [2, 8, 8, 3])
        grads1 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        grads2 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        grads3 = array_ops.placeholder(np.float32, [2, 8, 8, 5])
        weights = array_ops.placeholder(np.float32, [2, 2, 3, 5])
        vlr = array_ops.placeholder(np.float32, [])

        def conv_scaled_inplace(input_values, grads, lr):
          # yapf: disable
          return weights - nn_ops.conv2d_backprop_filter(
              input_values, filter_sizes, grads, strides=[1, 1, 1, 1
                                                  ], padding="SAME") * lr
          # yapf: enable

        result = (conv_scaled_inplace(input1, grads1, vlr) +
                  conv_scaled_inplace(input2, grads2, 0.1) +
                  conv_scaled_inplace(input3, grads3, 0.2))

      report = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      r = sess.run(
          result, {
              input1: np.ones([2, 8, 8, 3]),
              input2: np.ones([2, 8, 8, 3]),
              input3: np.ones([2, 8, 8, 3]),
              grads1: np.ones([2, 8, 8, 5]),
              grads2: np.ones([2, 8, 8, 5]),
              grads3: np.ones([2, 8, 8, 5]),
              weights: np.ones([2, 2, 3, 5]),
              vlr: 0.1,
          })
      # yapf: disable
      self.assertAllClose(r, [[[[-48.2, -48.2, -48.2, -48.2, -48.2],
                                [-48.2, -48.2, -48.2, -48.2, -48.2],
                                [-48.2, -48.2, -48.2, -48.2, -48.2],],
                               [[-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],],],
                              [[[-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],
                                [-41.8, -41.8, -41.8, -41.8, -41.8],],
                               [[-36.2, -36.2, -36.2, -36.2, -36.2],
                                [-36.2, -36.2, -36.2, -36.2, -36.2],
                                [-36.2, -36.2, -36.2, -36.2, -36.2],]]])
      # yapf: enable

      report.parse_log()

      # We still reuse the code even though only one conv is inplace.
      report.assert_compute_sets_matches('*Convolve', 1)

  def testConvolutionsWithBroadcast(self):
    with self.session() as sess:

      def model(device):
        with ops.device(device):
          x = array_ops.placeholder(np.float32, shape=[2])
          x_bcast = gen_array_ops.broadcast_to(x, shape=[2, 256, 256, 2])
          w_bcast = gen_array_ops.broadcast_to(x, shape=[2, 2, 2, 2])
          y = nn.conv2d(x_bcast, w_bcast, strides=1, padding="SAME", name="a")
          y = nn.conv2d(y, w_bcast, strides=1, padding="SAME", name="b")
          return sess.run(y, {x: np.ones(x.shape)})

      report = tu.ReportJSON(self, sess)

      report.reset()

      ipu_result = model("/device:IPU:0")
      cpu_result = model("cpu")
      self.assertAllClose(cpu_result, ipu_result)

      report.parse_log()

      report.assert_total_tile_memory(11336260)
      report.assert_max_tile_memory(9675)

      # Would fail if there were two convolutions in the graph
      ok = ['__seed*', 'a/convolution', 'Copy_']
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  googletest.main()

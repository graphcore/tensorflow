# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class ConvGraphCachingTest(xla_test.XLATestCase):
  def testConvolutionsMatch(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(y)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Would fail if there were two convolutions in the graph as they would be
      # called conv2d and conv2d_1
      ok = [
          '__seed*', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1', 'Copy_'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentTypes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(x)
          y = math_ops.cast(y, np.float16)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(y)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1', 'vs/Cast/convert.*/Cast',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentShapes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(x)
          y = array_ops.reshape(y, [1, 2, 8, 2])
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(y)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsDontMatchDifferentConvParams(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())(x)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              strides=(2, 1),
              kernel_initializer=init_ops.ones_initializer())(y)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_*weightsRearranged', 'host-exchange-local-copy-',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConvolutionsMatchFwdBwdWu(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')(x)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv2')(y)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv3')(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      ok = [
          '__seed*', 'host-exchange-local-copy-', 'Copy_',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropInput/fusion.*/WeightTranspose',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/DeltasPartialTranspose',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo'
      ]

  def testConvolutionsMatchFwdBwdWuVariableLR(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        lr = array_ops.placeholder(np.float32, shape=[])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')(x)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv2')(y)
          y = layers.Conv2D(
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv3')(y)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2]), lr: 0.1})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      # Fwd and BackpropInput should be shared
      # Weight transpose for BackpropInput should be present
      # Both BackpropFilter should be shared
      ok = [
          '__seed*',
          'host-exchange-local-copy-',
          'Copy_',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
          'vs/conv*/Conv2D/convolution*/Conv_1x1',
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

        def conv_scaled_inplace(input, grads, weights, lr):
          return weights - lr * nn_ops.conv2d_backprop_filter(
              input, filter_sizes, grads, strides=[1, 1, 1, 1], padding="SAME")

        result = (conv_scaled_inplace(input1, grads1, weights1, vlr) +
                  conv_scaled_inplace(input2, grads2, weights2, 0.1) +
                  conv_scaled_inplace(input3, grads3, weights3, 0.2))

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

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
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      self.assertEqual(tu.count_compute_sets_matching(cs_list, '*Convolve'), 1)

  def testConvolutionApplyNotInplace(self):
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

        def conv_scaled_inplace(input, grads, lr):
          return weights - nn_ops.conv2d_backprop_filter(
              input, filter_sizes, grads, strides=[1, 1, 1, 1],
              padding="SAME") * lr

        result = (conv_scaled_inplace(
            input1, grads1, vlr) + conv_scaled_inplace(input2, grads2, 0.1) +
                  conv_scaled_inplace(input3, grads3, 0.2))

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

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

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      self.assertEqual(tu.count_compute_sets_matching(cs_list, '*Convolve'), 2)


if __name__ == "__main__":
  googletest.main()

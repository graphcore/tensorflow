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

import tensorflow.compiler.plugin.poplar.tests.test_utils as tu
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import normalization as layers_norm
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class NormGraphCachingTest(xla_test.XLATestCase):
  def testBatchNormalizeInference(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Would fail if there were two batch norms in the graph
      ok = [
          '__seed*', 'Copy_',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference.*/'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBatchNormalizeInferenceDontMatchDifferentTypes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)
          y = math_ops.cast(y, np.float16)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference.*/',
          'convert.*/Cast', 'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization_1/FusedBatchNorm*/*'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBatchNormsDontMatchDifferentShapes(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)
          y = array_ops.reshape(y, [1, 2, 8, 2])
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()
      # Matches two convolutions
      ok = [
          '__seed*', 'Copy_', 'vs/conv2d/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference.*/',
          'vs/conv2d_1/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization_1/FusedBatchNorm*/batch-norm-inference.*/'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBatchNormsMatchFwdBwd(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')
          y = layers_norm.batch_normalization(y, fused=True, training=True)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv2')
          y = layers_norm.batch_normalization(y, fused=True, training=True)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv3')
          y = layers_norm.batch_normalization(y, fused=True, training=True)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # One BN for forwards and one BN for grad
      # (note that we don't cache gradient application)
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy*',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-training.*/',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/batch_normalization_2/FusedBatchNorm*_grad/FusedBatchNormGrad*/batch-norm-grad.*/',
          'GradientDescent/update_vs/batch_normalization/',
          'GradientDescent/update_vs/batch_normalization_1/',
          'GradientDescent/update_vs/batch_normalization_2/',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Transpose',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropInput/fusion/*Transpose',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testGroupNormalizeInference(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          gamma = constant_op.constant([0.5, 0.5], np.float32)
          beta = constant_op.constant([0.5, 0.5], np.float32)
          mean = constant_op.constant([0.5, 0.5], np.float32)
          inv_std_dev = constant_op.constant([0.5, 0.5], np.float32)
          y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                       gamma=gamma,
                                                       beta=beta,
                                                       mean=mean,
                                                       inv_std_dev=inv_std_dev,
                                                       data_format="NHWC",
                                                       epsilon=0.0015,
                                                       num_groups=2)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                       gamma=gamma,
                                                       beta=beta,
                                                       mean=mean,
                                                       inv_std_dev=inv_std_dev,
                                                       data_format="NHWC",
                                                       epsilon=0.0015,
                                                       num_groups=2)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Would fail if there were two batch norms in the graph
      ok = [
          '__seed*', 'Copy_',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
          'vs/PopnnGroupNormInference/custom-call*/'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testGroupNormalizeInferenceAndStatistics(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          gamma = constant_op.constant([0.5, 0.5], np.float32)
          beta = constant_op.constant([0.5, 0.5], np.float32)
          mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
              inputs=y, data_format="NHWC", epsilon=0.0015, num_groups=2)
          y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                       gamma=gamma,
                                                       beta=beta,
                                                       mean=mean,
                                                       inv_std_dev=inv_std_dev,
                                                       data_format="NHWC",
                                                       epsilon=0.0015,
                                                       num_groups=2)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
              inputs=y, data_format="NHWC", epsilon=0.0015, num_groups=2)
          y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                       gamma=gamma,
                                                       beta=beta,
                                                       mean=mean,
                                                       inv_std_dev=inv_std_dev,
                                                       data_format="NHWC",
                                                       epsilon=0.0015,
                                                       num_groups=2)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Would fail if there were two batch norms in the graph
      ok = [
          '__seed*', 'Copy_',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
          'vs/PopnnGroupNormStatistics/custom-call*/',
          'vs/PopnnGroupNormInference/custom-call*/'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testBatchNormAndGroupNormalizeMixedInference(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          gamma = constant_op.constant([0.5, 0.5], np.float32)
          beta = constant_op.constant([0.5, 0.5], np.float32)
          mean = constant_op.constant([0.5, 0.5], np.float32)
          inv_std_dev = constant_op.constant([0.5, 0.5], np.float32)
          y = gen_popnn_ops.popnn_group_norm_inference(inputs=y,
                                                       gamma=gamma,
                                                       beta=beta,
                                                       mean=mean,
                                                       inv_std_dev=inv_std_dev,
                                                       data_format="NHWC",
                                                       epsilon=0.0015,
                                                       num_groups=2)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer())
          y = layers_norm.batch_normalization(y, fused=True)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Would fail if there were two batch norms in the graph
      ok = [
          '__seed*', 'Copy_',
          'vs/conv2d/Conv2D/convolution.*/Conv_1x1/Convolve',
          'vs/PopnnGroupNormInference/custom-call*/',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-inference.*/'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testGroupNormsMatchFwdBwd(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(
              x,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')
          gamma = constant_op.constant([0.5, 0.5], np.float32)
          beta = constant_op.constant([0.5, 0.5], np.float32)
          y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                            gamma=gamma,
                                                            beta=beta,
                                                            data_format="NHWC",
                                                            epsilon=0.0015,
                                                            num_groups=2)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv2')
          y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                            gamma=gamma,
                                                            beta=beta,
                                                            data_format="NHWC",
                                                            epsilon=0.0015,
                                                            num_groups=2)
          y = convolutional.conv2d(
              y,
              2,
              1,
              use_bias=False,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv3')
          y, _, _ = gen_popnn_ops.popnn_group_norm_training(inputs=y,
                                                            gamma=gamma,
                                                            beta=beta,
                                                            data_format="NHWC",
                                                            epsilon=0.0015,
                                                            num_groups=2)

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # One GN for forwards and one GN for grad
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          'Copy_',
          'vs/conv1/Conv2D/convolution*/Conv_1x1/Convolve',
          'vs/PopnnGroupNormTraining/custom-call*/Norm',
          'vs/PopnnGroupNormTraining/custom-call*/iStdDev',
          'vs/PopnnGroupNormTraining/custom-call*/Whiten',
          'Sum/reduce.*/*/Reduce',
          'gradients/vs/PopnnGroupNormTraining_2_grad/PopnnGroupNormGrad/custom-call*/',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion.*',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropInput/fusion/*Transpose',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)

  def testNormCacheConstants(self):
    with self.session() as sess:

      def model(x, y, z):
        scale = gen_array_ops.broadcast_to(z, shape=[65536])
        offset = scale
        b_mean, b_var = nn.moments(x, [0, 1, 2], name='moments')
        b_mean_32 = math_ops.cast(b_mean, np.float32)
        b_var_32 = math_ops.cast(b_var, np.float32)
        a = nn.fused_batch_norm(x,
                                scale,
                                offset,
                                b_mean_32,
                                b_var_32,
                                1e-3,
                                is_training=False,
                                name="a")
        b = nn.fused_batch_norm(y,
                                scale,
                                offset,
                                b_mean_32,
                                b_var_32,
                                1e-3,
                                is_training=False,
                                name="b")

        return a[0] + b[0]

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float16, [1, 1, 1, 65536], name="x")
        y = array_ops.placeholder(np.float16, [1, 1, 1, 65536], name="y")
        z = array_ops.placeholder(np.float32, shape=[1])

      with ops.device("/device:IPU:0"):
        res = ipu_compiler.compile(model, inputs=[x, y, z])

      report = ReportJSON(self, sess)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())

      report.reset()

      r = sess.run(res, {x: np.ones(x.shape), y: np.ones(y.shape), z: [1.0]})
      self.assertAllClose(r[0], np.full(r[0].shape, 2))

      report.parse_log()

      report.assert_total_tile_memory(2744684)
      report.assert_max_tile_memory(2683)

      # Would fail if there were two batch norms in the graph
      ok = [
          '__seed*',
          'Copy_',
          'Cast',
          'moments',
          'a/batch-norm-inference',
          'a/batch-norm-inference*/Multiply',
          'add/add*/Add',
      ]
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  googletest.main()

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
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.layers import normalization as layers_norm


class NormGraphCachingTest(xla_test.XLATestCase):
  def testBatchNormsMatchFwdBwdSomeOnShard0SomeOnShard1(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          with ipu.scopes.ipu_shard(0):
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

          with ipu.scopes.ipu_shard(1):
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

      report = tu.ReportJSON(self, sess, sharded=True)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run([train, loss], {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Two BN for forwards (on shards 0 and 1) and two BN for grad
      # (note that we don't cache gradient application)
      # pylint: disable=line-too-long
      ok = [
          '__seed*',
          '*OnTileCopy*',
          'Copy_',
          'vs/conv1/Conv2D/convolution.*/Conv_1x1',
          'vs/conv3/Conv2D/convolution.*/Conv_1x1',
          'vs/batch_normalization/FusedBatchNorm*/batch-norm-training.*/',
          'vs/batch_normalization_2/FusedBatchNorm*/batch-norm-training.*/',
          'Sum/reduce.*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
          'Sum/reduce.*/ReduceFinalStage/IntermediateToOutput/Reduce',
          'gradients/vs/batch_normalization_2/FusedBatchNorm*_grad/FusedBatchNormGrad*/batch-norm-grad.*/',
          'gradients/vs/batch_normalization_1/FusedBatchNorm*_grad/FusedBatchNormGrad*/batch-norm-grad.*/',
          'GradientDescent/update_vs/batch_normalization/',
          'GradientDescent/update_vs/batch_normalization_1/',
          'GradientDescent/update_vs/batch_normalization_2/',
          'gradients/vs/conv*/Conv2D_grad/Conv2DBackpropFilter/fusion*/AddTo*',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropFilter/fusion*/Conv_4x4',
          'gradients/vs/conv3/Conv2D_grad/Conv2DBackpropInput/weights-transpose-chans-flip-x-y*/WeightsTransposeChansFlipXY/WeightsTranspose',
          'gradients/vs/conv2/Conv2D_grad/Conv2DBackpropInput/weights-transpose-chans-flip-x-y*/WeightsTransposeChansFlipXY/WeightsTranspose',
          'gradients/vs/conv1/Conv2D_grad/Conv2DBackpropFilter/fusion.*/Conv_4x4',
          'gradients/vs/conv1/Conv2D_grad/Conv2DBackpropFilter/fusion.*/AddTo',
      ]
      # pylint: enable=line-too-long
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  googletest.main()

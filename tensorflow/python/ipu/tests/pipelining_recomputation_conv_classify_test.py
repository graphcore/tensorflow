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

import numpy as np

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import pipelining_ops
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0

# Various graph constructor helpers


def fc(x, num_units_out):
  return layers.Dense(num_units_out,
                      kernel_initializer=init_ops.constant_initializer(0.1),
                      bias_initializer=init_ops.constant_initializer(0.0))(x)


class PipeliningRecomputationConvClassifyTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testTwoMatMuls(self):
    # Check that we get all classifications for a simple conv

    def stage1(x, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        x = fc(x, 16)
        x = nn.relu(x)
        x = fc(x, 48)
        x = nn.relu(x)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = fc(x, 48)
        x = nn.relu(x)
        x = fc(x, 100)
        x = nn.relu(x)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    # Run the pipeline twice.
    def model_pipeline(x, lr):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[x, lr],
                                     outfeed_queue=outfeed_queue,
                                     optimizer_function=optimizer_function)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 224])
      l = array_ops.placeholder(np.int32, shape=[1])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, l])

    tu.move_variable_initialization_to_cpu()
    outfeed_queue.dequeue()

    with tu.ipu_session() as sess:

      report = tu.ReportJSON(self, sess, pipelining=True, allow_recompute=True)
      sess.run(variables.global_variables_initializer())
      report.reset()
      sess.run(compiled_model_pipeline, {x: np.ones(x.shape), l: [1]})
      report.parse_log()

      # 2x matmul in 2 stages = 4x fwd x recomputation, 3x grads, 4x updates
      self.assertAllEqual(report.get_ml_type_counts(), [0, 8, 3, 4])

  @test_util.deprecated_graph_mode_only
  def testTwoParallelMatMuls(self):
    # Check that we get all classifications for a simple conv

    def stage1(x, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        a = fc(x, 48)
        a = nn.relu(a)
        b = fc(x, 48)
        b = nn.relu(b)
        return a + b, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        a = fc(x, 100)
        a = nn.relu(a)
        b = fc(x, 100)
        b = nn.relu(b)
        return a + b, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    # Run the pipeline twice.
    def model_pipeline(x, lr):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[x, lr],
                                     outfeed_queue=outfeed_queue,
                                     optimizer_function=optimizer_function)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 224])
      l = array_ops.placeholder(np.int32, shape=[1])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, l])

    tu.move_variable_initialization_to_cpu()
    outfeed_queue.dequeue()

    with tu.ipu_session() as sess:

      report = tu.ReportJSON(self, sess, pipelining=True, allow_recompute=True)
      sess.run(variables.global_variables_initializer())
      report.reset()
      sess.run(compiled_model_pipeline, {x: np.ones(x.shape), l: [1]})
      report.parse_log()

      # 2x matmul in 2 stages = 4x fwd x recomputation, 2x grads, 4x updates
      self.assertAllEqual(report.get_ml_type_counts(), [0, 8, 2, 4])


if __name__ == "__main__":
  googletest.main()

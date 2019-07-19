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
import numpy as np

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.ops import pipelining_ops_grad
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.compiler.xla import xla


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class PipeliningTest(test_util.TensorFlowTestCase):
  def testPipelineStage(self):
    def stage(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())(x)
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())(y)
        loss = math_ops.reduce_sum(y)
        return loss

    def my_net(x):
      # Compute with and without the pipeline stage.
      l = pipelining_ops._pipeline_stage(stage, [x]) - stage(x)
      return l

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(result[0], np.zeros(result[0].shape))

  def testPipelineStageGradNoIntermediates(self):
    def stage(name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        w1 = variable_scope.get_variable(
            "w1",
            shape=[10],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(1))
        return w1
    def my_net():
      l_pipeline = pipelining_ops._pipeline_stage(stage, [])
      optimizer1 = gradient_descent.GradientDescentOptimizer(0.1)
      pipeline_stage_vars = variables.trainable_variables()
      train_pipeline = optimizer1.minimize(l_pipeline)

      # Compute the same without the pipeline stage API.
      l = stage("non_pipeline")
      optimizer2 = gradient_descent.GradientDescentOptimizer(0.1)
      # Get the new variables.
      non_pipeline_stage_vars = [x for x in variables.trainable_variables() if x not in pipeline_stage_vars]
      train = optimizer2.minimize(l)

      # Subtract the loss and all the variables to make sure we get the same results.
      with ops.control_dependencies([train_pipeline, train]):
        diffs = [x - y for x,y in zip(pipeline_stage_vars, non_pipeline_stage_vars)]
        diffs.append(l_pipeline - l)
        return diffs

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r)
      for t in result:
        self.assertAllClose(t, np.zeros(t.shape))

  def testPipelineStageGradIntermediates(self):
    def stage(x, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer())(x)
        y = layers.MaxPooling2D(2, 1, "VALID")(y)
        return math_ops.reduce_sum(y)

    def my_net(x):
      l_pipeline = pipelining_ops._pipeline_stage(stage, [x])
      optimizer1 = gradient_descent.GradientDescentOptimizer(0.1)
      pipeline_stage_vars = variables.trainable_variables()
      train_pipeline = optimizer1.minimize(l_pipeline)

      # Compute the same without the pipeline stage API.
      l = stage(x, "non_pipeline")
      optimizer2 = gradient_descent.GradientDescentOptimizer(0.1)
      # Get the new variables.
      non_pipeline_stage_vars = [x for x in variables.trainable_variables() if x not in pipeline_stage_vars]
      train = optimizer2.minimize(l)

      # Subtract the loss and all the variables to make sure we get the same results.
      with ops.control_dependencies([train_pipeline, train]):
        diffs = [x - y for x,y in zip(pipeline_stage_vars, non_pipeline_stage_vars)]
        diffs.append(l_pipeline - l)
        return diffs

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {x: np.ones([1, 4, 4, 2])})
      for t in result:
        self.assertAllClose(t, np.zeros(t.shape))

  def testPipelineStageWithInfeeds(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return a, b

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def stage(a, b, name=None):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv1')(a)
      return math_ops.reduce_sum(y + b)

    def my_net():
      def body(loss):
        return loss + pipelining_ops._pipeline_stage(stage, [], infeed_queue=infeed_queue)
      r = loops.repeat(10, body, [0.0])
      return r

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      result = sess.run(r)
      self.assertAllClose(result, [6400.0])

  def testPipelineStageWithInfeedsKwargs(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def stage(c, name=None, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv1')(kwargs["a"])
      return math_ops.reduce_sum(y + kwargs["b"]) + c

    def my_net():
      def body(loss):
        return loss + pipelining_ops._pipeline_stage(stage, [10.0], infeed_queue=infeed_queue)
      r = loops.repeat(10, body, [0.0])
      return r

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      result = sess.run(r)
      self.assertAllClose(result, [6500.0])

  def testPipeline(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())(x)
        return y
    def stage2(x):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())(x)
        loss = math_ops.reduce_sum(y)
        return loss

    def my_net(x):
      # Compute with and without the pipeline stage.
      l = pipelining_ops.pipeline([stage1, stage2], [x]) - stage2(stage1(x))
      return l

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(result[0], np.zeros(result[0].shape))

  def testPipelineWithInfeedsKwargs(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def stage1(c, name=None, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer(),
            name='conv1')(kwargs["a"])
        return y + kwargs["b"], c
    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def my_net():
      def body(loss):
        return loss + pipelining_ops.pipeline([stage1, stage2], [10.0],
                                              infeed_queue=infeed_queue)
      r = loops.repeat(10, body, [0.0])
      return r

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      result = sess.run(r)
      self.assertAllClose(result, [6500.0])

  def testPipelineGradIntermediates(self):
    def stage1(x, name=None):
      name = "stage1_vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer())(x)
        return y
    def stage2(x, name=None):
      name = "stage2_vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer())(x)
        y = layers.MaxPooling2D(2, 1, "VALID")(y)
      return math_ops.reduce_sum(y)

    def my_net(x):
      l_pipeline = pipelining_ops.pipeline([stage1, stage2], [x])
      optimizer1 = gradient_descent.GradientDescentOptimizer(0.1)
      pipeline_stage_vars = variables.trainable_variables()
      train_pipeline = optimizer1.minimize(l_pipeline)

      # Compute the same without the pipeline stage API.
      l = stage2(stage1(x, "non_pipeline"), "non_pipeline")
      optimizer2 = gradient_descent.GradientDescentOptimizer(0.1)
      # Get the new variables.
      non_pipeline_stage_vars = [x for x in variables.trainable_variables() if x not in pipeline_stage_vars]
      train = optimizer2.minimize(l)

      # Subtract the loss and all the variables to make sure we get the same results.
      with ops.control_dependencies([train_pipeline, train]):
        diffs = [x - y for x,y in zip(pipeline_stage_vars, non_pipeline_stage_vars)]
        diffs.append(l_pipeline - l)
        return diffs

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {x: np.ones([1, 4, 4, 2])})
      for t in result:
        self.assertAllClose(t, np.zeros(t.shape))

  def testPipelineOnlyOneStage(self):
    def stage1(x, name=None):
      return x

    def my_net(x):
      return pipelining_ops.pipeline([stage1], [x])

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(
        ValueError, 'Pipeline requires at least one stage.' ):
        r = xla.compile(my_net, inputs=[x])

if __name__ == "__main__":
  googletest.main()

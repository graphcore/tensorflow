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
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.ops import pipelining_ops_grad
from tensorflow.python.ipu.optimizers import map_gradient_optimizer
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


def _run_no_pipeline(stages, inputs=None, optimizer_stage=None):
  outputs = inputs if inputs else []
  for stage in stages:
    outputs = stage(
        *pipelining_ops._convert_to_list(outputs), name="_no_pipeline")
  if optimizer_stage:
    outputs = optimizer_stage(*pipelining_ops._convert_to_list(outputs))
  return outputs


class PipeliningTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedInference(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      loss = math_ops.reduce_sum(x)
      return loss

    def my_net(x):
      return pipelining_ops.pipeline([stage1, stage2], [x])

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(
          ValueError, 'The last computational stage has tensor outputs'):
        r = ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedWithOutputsTraining(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      y = layers.Conv2D(
          2,
          1,
          use_bias=True,
          bias_initializer=init_ops.ones_initializer(),
          kernel_initializer=init_ops.ones_initializer())(x)
      loss = math_ops.reduce_sum(y)
      return loss

    def optimizer_stage(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)
      return loss, opt

    def my_net(x):
      return pipelining_ops.pipeline([stage1, stage2], [x],
                                     optimizer_stage=optimizer_stage)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError,
                                   'The optimizer_stage has tensor outputs'):
        r = ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineWithInfeedsKwargs(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

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

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2], [c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    tu.configure_ipu_system()

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[410.01]])

  @test_util.deprecated_graph_mode_only
  def testPipelineGradIntermediates(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with ops.device('cpu'):
      lr = array_ops.placeholder(np.float32, shape=[])

    def stage1(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer())(x)
        return y, lr

    def stage2(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer(),
            name="stage2")(x)
        return y, lr

    def stage3(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            bias_initializer=init_ops.ones_initializer(),
            kernel_initializer=init_ops.ones_initializer())(x)
        y = layers.MaxPooling2D(2, 1, "VALID")(y)
      return math_ops.reduce_sum(y), lr

    def optimizer_stage(loss, lr):
      opt = gradient_descent.GradientDescentOptimizer(lr)

      grads = gradients_impl.gradients(loss, variables.trainable_variables())
      grads = list(zip(grads, variables.trainable_variables()))
      grads = [(grad + (0.1 * var), var) if 'stage2' not in var.name else
               (grad, var) for grad, var in grads]
      grads = [(clip_ops.clip_by_value(grad, -1., 1.), var)
               for grad, var in grads]

      return loss, opt.apply_gradients(grads_and_vars=grads)

    def model_pipeline(x, lr):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     inputs=[x, lr],
                                     outfeed_queue=outfeed_queue,
                                     optimizer_stage=optimizer_stage)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      lr = array_ops.placeholder(np.float32, shape=[])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(
          model_pipeline, inputs=[x, lr])

    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(compiled_model_pipeline, {x: np.ones(x.shape), lr: 0.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[270.0]])

  @test_util.deprecated_graph_mode_only
  def testIllegalCapture(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with ops.device('cpu'):
      y = array_ops.placeholder(np.float32, shape=[])

    def stage1(x):
      return x * y

    def stage2(x):
      return x

    def model_pipeline(x):
      return pipelining_ops.pipeline([stage1, stage2],
                                     inputs=[x],
                                     outfeed_queue=outfeed_queue)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError, 'Trying to capture the tensor'):
        ipu_compiler.compile(model_pipeline, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineOnlyOneStage(self):
    def stage1(x, name=None):
      return x

    def my_net(x):
      return pipelining_ops.pipeline([stage1], [x])

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError,
                                   'Pipeline requires at least two'):
        r = ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testDuplicateInputsOutputs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(x, y):
      return x, y, y, x

    # The above should be optimised to a single copy for each duplicate output.
    def stage2(x1, y1, y2, x2):
      return x2, y2

    def model_pipeline(x, y):
      return pipelining_ops.pipeline([stage1, stage2],
                                     inputs=[x, y],
                                     outfeed_queue=outfeed_queue)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[1, 2])

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(
          model_pipeline, inputs=[x, y])
    #TODO(T10784) test how many IPU copies are here once we insert IPU copies.
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(compiled_model_pipeline, {
          x: np.ones(x.shape),
          y: np.ones(y.shape)
      })
      output = sess.run(outfeed_op)
      self.assertAllClose(output[0][0], np.ones(x.shape))
      self.assertAllClose(output[1][0], np.ones(y.shape))


if __name__ == "__main__":
  googletest.main()

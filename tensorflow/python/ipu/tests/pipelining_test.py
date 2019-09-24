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

import numpy as np

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import gradient_accumulation_optimizer
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import normalization_ops
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import scopes


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class _PipelineTester(object):
  @staticmethod
  def _cpu_with_grad_accum(session, stages, inputs, input_values, repeat_count,
                           pipeline_depth, dataset, optimizer):

    with variable_scope.variable_scope("cpu", use_resource=True):

      def pipeline(*args):
        iterator = dataset.make_one_shot_iterator()
        next_example, next_label = iterator.get_next()
        outputs = pipelining_ops._convert_to_list(args)
        outputs.append(next_example)
        outputs.append(next_label)
        for stage in stages:
          outputs = stage(*pipelining_ops._convert_to_list(outputs))
        return outputs

      loss = pipeline(*inputs)

      trainable_variables = variables.trainable_variables()
      accum_vars = [
          standard_ops.Variable(array_ops.zeros_like(var.initialized_value()),
                                trainable=False) for var in trainable_variables
      ]
      zero_ops = [var.assign(array_ops.zeros_like(var)) for var in accum_vars]
      grads = optimizer.compute_gradients(loss, trainable_variables)
      accum_ops = [
          accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)
      ]
      train_step = optimizer.apply_gradients([(accum_vars[i], gv[1])
                                              for i, gv in enumerate(grads)])

    session.run(variables.global_variables_initializer())
    losses = []
    with ops.device("cpu"):
      for _ in range(repeat_count):
        session.run(zero_ops)
        for _ in range(pipeline_depth):
          l, _ = session.run([loss, accum_ops],
                             feed_dict=dict(zip(inputs, input_values)))
          losses.append(l)
        # Run the train_step ops to update the weights based on accumulated
        # gradients
        session.run(train_step)
    return losses

  @staticmethod
  def _sharded_on_ipu(session, stages, inputs, input_values, repeat_count,
                      pipeline_depth, dataset, optimizer):

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with variable_scope.variable_scope("ipu_sharded", use_resource=True):

      def pipeline(*args):
        outputs = args
        for i, stage in enumerate(stages):
          with scopes.ipu_shard(i):
            outputs = stage(*pipelining_ops._convert_to_list(outputs))
        loss = outputs
        enqueue_op = outfeed_queue.enqueue(loss)
        opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
            optimizer, pipeline_depth)
        outs = list(args[:len(args) - infeed_queue.number_of_tuple_elements])
        outs.append(enqueue_op)
        outs.append(opt.minimize(loss))
        return outs

      def my_net(*args):
        return loops.repeat(pipeline_depth,
                            pipeline,
                            inputs=args,
                            infeed_queue=infeed_queue)

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

    outfeed_op = outfeed_queue.dequeue()
    tu.configure_ipu_system(pipelining=True, text_report=False)
    tu.move_variable_initialization_to_cpu()

    session.run(variables.global_variables_initializer())
    session.run(infeed_queue.initializer)
    for _ in range(repeat_count):
      session.run(compiled_model_pipeline,
                  feed_dict=dict(zip(inputs, input_values)))
    return session.run(outfeed_op)

  @staticmethod
  def _pipeline_on_ipu(session, stages, inputs, input_values, repeat_count,
                       pipeline_depth, dataset, optimizer, test_wrapper):

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with variable_scope.variable_scope("ipu", use_resource=True):

      def optimizer_stage(loss):
        opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
            optimizer, pipeline_depth)
        return loss, opt.minimize(loss)

      def my_net(*args):
        return pipelining_ops.pipeline(stages,
                                       pipeline_depth,
                                       repeat_count=repeat_count,
                                       inputs=args,
                                       optimizer_stage=optimizer_stage,
                                       infeed_queue=infeed_queue,
                                       outfeed_queue=outfeed_queue)

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

    outfeed_op = outfeed_queue.dequeue()
    report = tu.ReportJSON(test_wrapper,
                           session,
                           pipelining=True,
                           io_trace=False)
    tu.move_variable_initialization_to_cpu()

    session.run(variables.global_variables_initializer())
    session.run(infeed_queue.initializer)
    session.run(compiled_model_pipeline,
                feed_dict=dict(zip(inputs, input_values)))
    out = session.run(outfeed_op)[0]
    report.parse_log()
    report.assert_pipeline_stages_on_expected_ipu(range(len(stages)))
    return out

  @staticmethod
  def compare_pipeline_to_cpu(session, stages, inputs, input_values,
                              repeat_count, pipeline_depth, dataset_fn,
                              optimizer, test_wrapper):
    cpu_losses = _PipelineTester._cpu_with_grad_accum(session, stages, inputs,
                                                      input_values,
                                                      repeat_count,
                                                      pipeline_depth,
                                                      dataset_fn(), optimizer)
    pipeline_losses = _PipelineTester._pipeline_on_ipu(session, stages, inputs,
                                                       input_values,
                                                       repeat_count,
                                                       pipeline_depth,
                                                       dataset_fn(), optimizer,
                                                       test_wrapper)
    test_wrapper.assertAllClose(cpu_losses, pipeline_losses)

  @staticmethod
  def compare_pipeline_to_sharding(session, stages, inputs, input_values,
                                   repeat_count, pipeline_depth, dataset_fn,
                                   optimizer, test_wrapper):
    pipeline_losses = _PipelineTester._pipeline_on_ipu(session, stages, inputs,
                                                       input_values,
                                                       repeat_count,
                                                       pipeline_depth,
                                                       dataset_fn(), optimizer,
                                                       test_wrapper)
    sharded_losses = _PipelineTester._sharded_on_ipu(session, stages, inputs,
                                                     input_values,
                                                     repeat_count,
                                                     pipeline_depth,
                                                     dataset_fn(), optimizer)
    test_wrapper.assertAllClose(sharded_losses, pipeline_losses)


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
      return pipelining_ops.pipeline([stage1, stage2], 10, inputs=[x])

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(
          ValueError, 'The last computational stage has tensor outputs'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedWithOutputsTraining(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      y = layers.Conv2D(2,
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
      return pipelining_ops.pipeline([stage1, stage2],
                                     10,
                                     inputs=[x],
                                     optimizer_stage=optimizer_stage)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError,
                                   'The optimizer_stage has tensor outputs'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineIterationsNotMultiple(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     10,
                                     inputs=[c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegexp(
          errors.FailedPreconditionError,
          'The pipeline depth of the pipeline must be a multiple of 3'):
        sess.run(r, {c: 10.01})

  @test_util.deprecated_graph_mode_only
  def testPipelineMinIterations(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     3,
                                     inputs=[c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegexp(
          errors.FailedPreconditionError,
          'The pipeline depth of the pipeline must be at least 6'):
        sess.run(r, {c: 10.01})

  @test_util.deprecated_graph_mode_only
  def testPipelineInvalidDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    # Wrong type:
    with self.assertRaisesRegexp(
        ValueError, 'device_mapping argument needs to be a list or a tuple'):
      pipelining_ops.pipeline([stage1, stage2, stage3],
                              3,
                              inputs=[c],
                              infeed_queue=infeed_queue,
                              outfeed_queue=outfeed_queue,
                              device_mapping=1)

    # Too many values:
    with self.assertRaisesRegexp(ValueError,
                                 'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline([stage1, stage2, stage3],
                              3,
                              inputs=[c],
                              infeed_queue=infeed_queue,
                              outfeed_queue=outfeed_queue,
                              device_mapping=list(range(4)))

    # Not enough values:
    with self.assertRaisesRegexp(ValueError,
                                 'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline([stage1, stage2, stage3],
                              3,
                              inputs=[c],
                              infeed_queue=infeed_queue,
                              outfeed_queue=outfeed_queue,
                              device_mapping=tuple(range(1)))

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())
    device_mapping = [2, 0, 1]

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue,
                                     device_mapping=device_mapping)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, pipelining=True, io_trace=False)
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(device_mapping)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMappingSameIpu(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())
    device_mapping = [2, 2, 2]

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue,
                                     device_mapping=device_mapping)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, pipelining=True, io_trace=False)
      report.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(device_mapping)

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

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[c],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, pipelining=True, io_trace=False)
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(range(3))

  @test_util.deprecated_graph_mode_only
  def testPipelineGradIntermediates(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with ops.device('cpu'):
      lr = array_ops.placeholder(np.float32, shape=[])

    def stage1(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          bias_initializer=init_ops.ones_initializer(),
                          kernel_initializer=init_ops.ones_initializer())(x)
        return y, lr

    def stage2(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          bias_initializer=init_ops.ones_initializer(),
                          kernel_initializer=init_ops.ones_initializer(),
                          name="stage2")(x)
        return y, lr

    def stage3(x, lr, name=None):
      name = "vs" + name if name else ""
      with variable_scope.variable_scope(name, use_resource=True):
        y = layers.Conv2D(2,
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
      opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
          opt, 6)

      return loss, opt.apply_gradients(grads_and_vars=grads)

    def model_pipeline(x, lr):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[x, lr],
                                     outfeed_queue=outfeed_queue,
                                     optimizer_stage=optimizer_stage)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      lr = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, lr])

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, pipelining=True, io_trace=False)
      sess.run(variables.global_variables_initializer())
      sess.run(compiled_model_pipeline, {x: np.ones(x.shape), lr: 0.01})
      losses_pipeline = sess.run(outfeed_op)
      # Note that the pipeline always takes the same input - see how the
      # loss is the same for first 6 executions (gradient accumulation), then
      # there is one stage which uses stale weights, and the remaining stages
      # return the same value.
      self.assertAllClose(losses_pipeline, [[
          270., 270., 270., 270., 270., 270., 253.79999, 239.5872, 239.5872,
          239.5872, 239.5872, 239.5872
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(range(3))

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
                                     10,
                                     inputs=[x],
                                     outfeed_queue=outfeed_queue)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[])

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError, 'Trying to capture the tensor'):
        ipu_compiler.compile(model_pipeline, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineOnlyOneStage(self):
    def stage1(x):
      return x

    def my_net(x):
      return pipelining_ops.pipeline([stage1], 10, inputs=[x])

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegexp(ValueError,
                                   'Pipeline requires at least two'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testDuplicateInputsOutputs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(x, y):
      return x, y, y, x

    # The above should be optimised to a single copy for each duplicate output.
    def stage2(x1, y1, y2, x2):
      return x1, y1, y2, x2

    # Same for this stage
    def stage3(_x1, _y1, y2, x2):
      return x2, y2

    def model_pipeline(x, y):
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     12,
                                     inputs=[x, y],
                                     outfeed_queue=outfeed_queue)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[1, 2])

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, y])
    #TODO(T10784) test how many IPU copies are here once we insert IPU copies.
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(compiled_model_pipeline, {
          x: np.ones(x.shape),
          y: np.ones(y.shape)
      })
      output = sess.run(outfeed_op)
      for i in range(12):
        self.assertAllClose(output[0][i], np.ones(x.shape))
        self.assertAllClose(output[1][i], np.ones(y.shape))

  @test_util.deprecated_graph_mode_only
  def testPipelineWithStagesWithConstants(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      idx = value[0][0][0][0]
      return {"a": a, "b": b, "idx": idx}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def stage1(c, **kwargs):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(kwargs["a"])
      y = normalization_ops.group_norm(y)
      return y + kwargs["b"], c, kwargs["idx"]

    def stage2(x, c, idx):
      return x, c, idx

    def stage3(x, c, idx):
      return layers.Dense(
          2,
          kernel_initializer=init_ops.ones_initializer(),
          bias_initializer=init_ops.ones_initializer())(x), c, idx

    def stage4(x, c, idx):
      return math_ops.reduce_sum(
          layers.Dense(
              2,
              kernel_initializer=init_ops.ones_initializer(),
              bias_initializer=init_ops.ones_initializer())(x)) + c, idx

    def optimizer_stage(loss, idx):
      opt = gradient_descent.GradientDescentOptimizer(0.01)

      grads = gradients_impl.gradients(loss, variables.trainable_variables())
      grads = list(zip(grads, variables.trainable_variables()))
      grads = [(clip_ops.clip_by_value(grad, -1., 1.), var)
               for grad, var in grads]
      opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
          opt, 12)
      return loss, idx, opt.apply_gradients(grads_and_vars=grads)

    # Run the pipeline twice.
    def my_net(c):
      return pipelining_ops.pipeline([stage1, stage2, stage3, stage4],
                                     12,
                                     inputs=[c],
                                     optimizer_stage=optimizer_stage,
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    tu.configure_ipu_system(pipelining=True, text_report=False)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      # The values have been verified and compared against running the same
      # graph but sharded with gradient accumulation for 12 mini batches.
      self.assertAllClose(losses_pipeline[0], [
          1546.01, 1802.01, 1738.01, 1674.01, 1930.01, 1546.01, 1802.01,
          1738.01, 1674.01, 1930.01, 1546.01, 1802.01, 1331.1415, 1281.5806,
          1479.8259, 1182.457, 1380.7043, 1331.1415, 1281.5806, 1479.8259,
          1182.457, 1380.7043, 1331.1415, 1281.5806
      ])
      self.assertAllClose(losses_pipeline[1], [
          0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4,
          1
      ])

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    pipeline_depth = 20
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x * 20, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return layers.Dense(
            2,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5))(x), c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(
            layers.Dense(2,
                         kernel_initializer=init_ops.constant_initializer(0.5),
                         bias_initializer=init_ops.constant_initializer(0.5))
            (x)) + c + label

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with self.test_session() as sess:
      _PipelineTester.compare_pipeline_to_cpu(sess,
                                              [stage1, stage2, stage3, stage4],
                                              [c], [10.01], repeat_count,
                                              pipeline_depth, dataset_fn,
                                              optimizer, self)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare2(self):
    # Resnet like network.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(100, shape=[4])
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    pipeline_depth = 18
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

    def block(name, first_stride, out_filters, count, x):

      for i in range(count):
        sc = x
        shape_in = x.shape
        stride = first_stride if (i == 0) else 1

        with variable_scope.variable_scope(name + "/" + str(i) + "/1"):
          x = conv(x, 3, stride, out_filters)
          x = nn.relu(x)

        with variable_scope.variable_scope(name + "/" + str(i) + "/2"):
          x = conv(x, 3, 1, out_filters)

          # shortcut
          if stride != 1:
            sc = array_ops.strided_slice(sc, [0, 0, 0, 0],
                                         sc.shape,
                                         strides=[1, stride, stride, 1])
          pad = int(x.shape[3] - shape_in[3])
          if pad != 0:
            sc = array_ops.pad(sc, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])

          x = nn.relu(x + sc)

      return x

    def fc(x, num_units_out):
      return layers.Dense(
          num_units_out,
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def max_pool(x, ksize=3, stride=2):
      return layers.MaxPooling2D(ksize, stride, padding='SAME')(x)

    def conv(x, ksize, stride, filters_out):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def stage1(img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        x = conv(img, 7, 2, 16)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = block("b", 2, 64, 1, x)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 100)
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    with self.test_session() as sess:
      _PipelineTester.compare_pipeline_to_sharding(sess,
                                                   [stage1, stage2, stage3],
                                                   [], [], repeat_count,
                                                   pipeline_depth, dataset_fn,
                                                   optimizer, self)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompare3(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    pipeline_depth = 20
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(idx, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            shape=[10, 1216],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(10.01),
            trainable=True)
        x = embedding_ops.embedding_lookup(embedding, idx)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        logits = math_ops.reduce_sum(x, axis=[-1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    with self.test_session() as sess:
      _PipelineTester.compare_pipeline_to_cpu(sess,
                                              [stage1, stage2, stage3, stage4],
                                              [], [], repeat_count,
                                              pipeline_depth, dataset_fn,
                                              optimizer, self)


if __name__ == "__main__":
  googletest.main()

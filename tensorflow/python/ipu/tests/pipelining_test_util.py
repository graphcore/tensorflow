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

import math

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.compat.v1 import data as compat_v1_data


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


def get_num_ipus(device_mapping):
  min_ipus = max(device_mapping) + 1
  return int(math.pow(2, math.ceil(math.log2(min_ipus))))


class PipelineTester(object):
  @staticmethod
  def _cpu_with_grad_accum(test_wrapper, stages, inputs_fn, input_values,
                           repeat_count, num_batches_to_accumulate, dataset_fn,
                           optimizer):

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      dataset = dataset_fn()
      inputs = inputs_fn()
      with variable_scope.variable_scope("cpu", use_resource=True,
                                         reuse=False):

        def pipeline(*args):
          # TF2 replacement for: iterator = dataset.make_one_shot_iterator()
          iterator = compat_v1_data.make_one_shot_iterator(dataset)
          next_example, next_label = iterator.get_next()
          outputs = functional_ops._convert_to_list(args)  # pylint: disable=W0212
          outputs.append(next_example)
          outputs.append(next_label)
          for stage in stages:
            outputs = stage(*functional_ops._convert_to_list(outputs))  # pylint: disable=W0212
          return outputs

        loss = pipeline(*inputs)

        trainable_variables = variables.trainable_variables()
        accum_vars = [
            standard_ops.Variable(array_ops.zeros_like(
                var.initialized_value()),
                                  trainable=False)
            for var in trainable_variables
        ]
        zero_ops = [
            var.assign(array_ops.zeros_like(var)) for var in accum_vars
        ]
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
          for _ in range(num_batches_to_accumulate):
            l, _ = session.run([loss, accum_ops],
                               feed_dict=dict(zip(inputs, input_values)))
            losses.append(l)
          # Run the train_step ops to update the weights based on accumulated
          # gradients
          session.run(train_step)
      return losses

  @staticmethod
  def _sharded_on_ipu(stages, inputs_fn, input_values, repeat_count,
                      num_batches_to_accumulate, dataset_fn, optimizer,
                      test_wrapper, recomp, device_mapping):

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      dataset = dataset_fn()
      inputs = inputs_fn()
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

      with variable_scope.variable_scope("ipu_sharded",
                                         use_resource=True,
                                         reuse=False):
        if device_mapping is None:
          device_mapping = range(len(stages))

        def pipeline(*args):
          outputs = args
          for i, stage in zip(device_mapping, stages):
            with scopes.ipu_shard(i):
              outputs = stage(*functional_ops._convert_to_list(outputs))  # pylint: disable=W0212
          loss = outputs
          enqueue_op = outfeed_queue.enqueue(loss)
          opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
              optimizer, num_batches_to_accumulate)
          outs = list(args[:len(args) - infeed_queue.number_of_tuple_elements])
          outs.append(enqueue_op)
          outs.append(opt.minimize(loss))
          return outs

        def my_net(*args):
          return loops.repeat(num_batches_to_accumulate,
                              pipeline,
                              inputs=args,
                              infeed_queue=infeed_queue)

      with ops.device("/device:IPU:0"):
        compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

      outfeed_op = outfeed_queue.dequeue()

      # Execution profiles of code with dynamic control flow are not supported on real HW
      profiling = utils.running_on_ipu_model()

      cfg = utils.create_ipu_config(profiling=profiling,
                                    profile_execution=profiling)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      num_ipus = get_num_ipus(device_mapping) if device_mapping else 4
      cfg = utils.auto_select_ipus(cfg, num_ipus)
      if recomp:
        cfg = utils.set_recomputation_options(cfg, allow_recompute=True)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      session.run(variables.global_variables_initializer())
      session.run(infeed_queue.initializer)
      for _ in range(repeat_count):
        session.run(compiled_model_pipeline,
                    feed_dict=dict(zip(inputs, input_values)))
      return session.run(outfeed_op)

  @staticmethod
  def pipeline_on_ipu(stages,
                      inputs_fn,
                      input_values,
                      repeat_count,
                      gradient_accumulation_count,
                      dataset_fn,
                      optimizer,
                      test_wrapper,
                      expected_max_tile_memory,
                      recomp,
                      schedule,
                      device_mapping=None,
                      batch_serialization_iterations=1):

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      dataset = dataset_fn()
      inputs = inputs_fn()
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

      with variable_scope.variable_scope("ipu", use_resource=True,
                                         reuse=False):

        def optimizer_function(loss):
          return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

        def my_net(*args):
          return pipelining_ops.pipeline(
              stages,
              gradient_accumulation_count,
              repeat_count=repeat_count,
              batch_serialization_iterations=batch_serialization_iterations,
              inputs=args,
              optimizer_function=optimizer_function,
              infeed_queue=infeed_queue,
              outfeed_queue=outfeed_queue,
              pipeline_schedule=schedule,
              device_mapping=device_mapping)

      with ops.device("/device:IPU:0"):
        compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

      # Execution profiles of code with dynamic control flow are not supported
      # on real HW.
      profiling = utils.running_on_ipu_model()
      cfg = utils.create_ipu_config(profiling=profiling,
                                    profile_execution=profiling)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      num_ipus = get_num_ipus(device_mapping) if device_mapping else 4
      cfg = utils.auto_select_ipus(cfg, num_ipus)
      if recomp:
        cfg = utils.set_recomputation_options(cfg, allow_recompute=True)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()
      report = tu.ReportJSON(test_wrapper, session, configure_device=False)

      session.run(variables.global_variables_initializer())
      session.run(infeed_queue.initializer)
      report.reset()
      session.run(compiled_model_pipeline,
                  feed_dict=dict(zip(inputs, input_values)))
      out = session.run(outfeed_op)[0]
      if profiling:
        report.parse_log()
        if not device_mapping:
          device_mapping = [
              i - (i % 4) + ((i % 4) if (i % 4) < 2 else 5 - (i % 4))
              for i in range(len(stages))
          ]
        report.assert_pipeline_stages_on_expected_ipu(device_mapping)
        report.assert_max_tile_memory(expected_max_tile_memory, tolerance=0.3)
      return out

  @staticmethod
  def compare_pipeline_to_cpu(stages,
                              inputs_fn,
                              input_values,
                              repeat_count,
                              gradient_accumulation_count,
                              dataset_fn,
                              optimizer,
                              test_wrapper,
                              expected_max_tile_memory,
                              recomp=False,
                              schedule=None,
                              device_mapping=None,
                              batch_serialization_iterations=1):

    if batch_serialization_iterations > 1:
      assert device_mapping is None
      device_mapping = [0] * len(stages)

    # Run pipeline_on_ipu before the CPU version as pipeline_on_ipu
    # will initialize the IPU which might be needed by the CPU path
    # to run some instructions which only exist on the IPU.
    pipeline_losses = PipelineTester.pipeline_on_ipu(
        stages, inputs_fn, input_values, repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer, test_wrapper,
        expected_max_tile_memory, recomp, schedule, device_mapping,
        batch_serialization_iterations)

    num_batches_to_accumulate = (gradient_accumulation_count *
                                 batch_serialization_iterations)
    cpu_losses = PipelineTester._cpu_with_grad_accum(
        test_wrapper, stages, inputs_fn, input_values, repeat_count,
        num_batches_to_accumulate, dataset_fn, optimizer)

    test_wrapper.assertAllClose(cpu_losses, pipeline_losses)

  @staticmethod
  def compare_pipeline_to_sharding(stages,
                                   inputs_fn,
                                   input_values,
                                   repeat_count,
                                   gradient_accumulation_count,
                                   dataset_fn,
                                   optimizer,
                                   test_wrapper,
                                   expected_max_tile_memory,
                                   recomp=False,
                                   schedule=None,
                                   device_mapping=None,
                                   batch_serialization_iterations=1):
    if batch_serialization_iterations > 1:
      assert device_mapping is None
      device_mapping = [0] * len(stages)

    pipeline_losses = PipelineTester.pipeline_on_ipu(
        stages, inputs_fn, input_values, repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer, test_wrapper,
        expected_max_tile_memory, recomp, schedule, device_mapping,
        batch_serialization_iterations)

    num_batches_to_accumulate = (gradient_accumulation_count *
                                 batch_serialization_iterations)
    sharded_losses = PipelineTester._sharded_on_ipu(stages, inputs_fn,
                                                    input_values, repeat_count,
                                                    num_batches_to_accumulate,
                                                    dataset_fn, optimizer,
                                                    test_wrapper, recomp,
                                                    device_mapping)
    test_wrapper.assertAllClose(sharded_losses, pipeline_losses)

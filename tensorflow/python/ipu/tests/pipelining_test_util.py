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
import pva

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
from tensorflow.python.ipu import cross_replica_optimizer
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ipu.utils import MergeRemoteBuffersBehaviour
from tensorflow.compat.v1 import data as compat_v1_data


def get_num_ipus(device_mapping):
  device_mapping = pipelining_ops._to_flat_list(device_mapping)  # pylint: disable=W0212
  min_ipus = max(device_mapping) + 1
  return int(math.pow(2, math.ceil(math.log2(min_ipus))))


def _get_vars(session, scope):
  # The reason for that is what looks like a bug in tensorflow slot variable creation code.
  # Instead of using variable name as an fully qualified name, it appends it to the current
  # scope. This ends up with scope/scope/var/slot naming.
  double_scope = scope + "/" + scope
  global_vars = variables.global_variables(scope=scope)
  values = session.run(global_vars)
  result = {}
  for var, value in zip(global_vars, values):
    name = var.name
    if name.startswith(double_scope):
      name = name[len(double_scope) + 1:]
    elif name.startswith(scope):
      name = name[len(scope) + 1:]
    result[name] = value
  return result


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
          nexts = iterator.get_next()
          outputs = functional_ops._convert_to_list(args)  # pylint: disable=W0212
          for n in nexts:
            outputs.append(n)
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
  def run_on_cpu(test_wrapper, stages, inputs_fn, input_values, repeat_count,
                 gradient_accumulation_count, dataset_fn, optimizer):
    return PipelineTester._cpu_with_grad_accum(test_wrapper, stages, inputs_fn,
                                               input_values, repeat_count,
                                               gradient_accumulation_count,
                                               dataset_fn, optimizer)

  @staticmethod
  def _sharded_on_ipu(
      stages,
      inputs_fn,
      input_values,
      repeat_count,
      num_batches_to_accumulate,
      dataset_fn,
      optimizer,
      test_wrapper,
      recomp,
      device_mapping,
      number_of_io_tiles=0,
      replication_factor=1,
      merge_remote_buffers=MergeRemoteBuffersBehaviour.IF_BENEFICIAL,
      replicated_optimizer_state_sharding=False,
      minimum_remote_tensor_size=128,
      return_vars=False):

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      dataset = dataset_fn()
      inputs = inputs_fn()
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      if device_mapping is None:
        device_mapping = range(len(stages))

      def pipeline(*args):
        outputs = args
        for i, stage in zip(device_mapping, stages):
          if isinstance(i, list):
            intermediates = []

            for k, s in zip(i, stage):
              with scopes.ipu_shard(k):
                intermediates.append(
                    s(*functional_ops._convert_to_list(outputs)))  # pylint: disable=W0212
            outputs = pipelining_ops._to_flat_list(intermediates)  # pylint: disable=W0212
          else:
            with scopes.ipu_shard(i):
              outputs = stage(*functional_ops._convert_to_list(outputs))  # pylint: disable=W0212
        enqueue_op = outfeed_queue.enqueue(outputs)
        outs = list(args[:len(args) - infeed_queue.number_of_tuple_elements])
        outs.append(enqueue_op)
        if optimizer:
          if replication_factor > 1:
            opt = \
              gradient_accumulation_optimizer.CrossReplicaGradientAccumulationOptimizerV2(# pylint: disable=line-too-long
                  optimizer,
                  num_batches_to_accumulate,
                  replicated_optimizer_state_sharding=replicated_optimizer_state_sharding) # pylint: disable=line-too-long
          else:
            opt = \
              gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
                  optimizer,
                  num_batches_to_accumulate,
                  replicated_optimizer_state_sharding=replicated_optimizer_state_sharding)# pylint: disable=line-too-long

          outs.append(opt.minimize(outputs))
        return outs

      def my_net(*args):
        with variable_scope.variable_scope("ipu_sharded",
                                           use_resource=True,
                                           reuse=False):
          return loops.repeat(num_batches_to_accumulate,
                              pipeline,
                              inputs=args,
                              infeed_queue=infeed_queue)

      with ops.device("/device:IPU:0"):
        compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

      outfeed_op = outfeed_queue.dequeue()

      # Execution profiles of code with dynamic control flow are not supported on real HW
      profiling = utils.running_on_ipu_model()

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = profiling  # pylint: disable=protected-access
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      if number_of_io_tiles > 0:
        cfg.io_tiles.num_io_tiles = number_of_io_tiles
        cfg.io_tiles.place_ops_on_io_tiles = True
      num_ipus = get_num_ipus(device_mapping) if device_mapping else 4
      num_ipus = num_ipus * replication_factor
      if tu.has_ci_ipus():
        tu.add_hw_ci_connection_options(cfg)
      cfg.auto_select_ipus = num_ipus
      if recomp:
        cfg.allow_recompute = True
      cfg.optimizations.merge_remote_buffers = merge_remote_buffers
      cfg.optimizations.minimum_remote_tensor_size = minimum_remote_tensor_size
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      session.run(variables.global_variables_initializer())
      session.run(infeed_queue.initializer)
      for _ in range(repeat_count):
        session.run(compiled_model_pipeline,
                    feed_dict=dict(zip(inputs, input_values)))
      outs = session.run(outfeed_op)
      if return_vars:
        return outs, _get_vars(session, "ipu_sharded")
      return outs

  @staticmethod
  def pipeline_on_ipu(
      stages,
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
      batch_serialization_iterations=1,
      recomputation_mode=None,
      number_of_io_tiles=0,
      return_report=False,
      replication_factor=1,
      offload_activations=None,
      merge_remote_buffers=MergeRemoteBuffersBehaviour.IF_BENEFICIAL,
      replicated_optimizer_state_sharding=False,
      minimum_remote_tensor_size=128,
      return_vars=False,
      ipu_id=None,
      process_count=None,
      process_index=None):

    g = ops.Graph()
    with g.as_default(), test_wrapper.test_session(graph=g) as session:
      dataset = dataset_fn()
      inputs = inputs_fn()
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def opt_fn(loss):
        global_replication_factor = replication_factor * (process_count or 1)
        if global_replication_factor > 1:
          opt = cross_replica_optimizer.CrossReplicaOptimizer(optimizer)
        else:
          opt = optimizer
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      optimizer_function = opt_fn if optimizer else None

      def my_net(*args):
        with variable_scope.variable_scope("ipu",
                                           use_resource=True,
                                           reuse=False):
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
              recomputation_mode=recomputation_mode,
              device_mapping=device_mapping,
              offload_activations=offload_activations,
              replicated_optimizer_state_sharding=
              replicated_optimizer_state_sharding)

      with ops.device("/device:IPU:0"):
        compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=inputs)

      # Execution profiles of code with dynamic control flow are not supported
      # on real HW.
      profiling = return_report and utils.running_on_ipu_model()
      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      if number_of_io_tiles > 0:
        cfg.io_tiles.num_io_tiles = number_of_io_tiles
        cfg.io_tiles.place_ops_on_io_tiles = True

      if profiling:
        cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
        report_helper = tu.ReportHelper()
        report_helper.set_autoreport_options(cfg)

      if ipu_id is None:
        num_ipus = get_num_ipus(device_mapping) if device_mapping else 4
        num_ipus = num_ipus * replication_factor
        cfg.auto_select_ipus = num_ipus
      else:
        cfg.select_ipus = [ipu_id]

      if process_count is not None:
        assert process_index is not None
        cfg.experimental.multi_replica_distribution.process_count = \
            process_count
        cfg.experimental.multi_replica_distribution.process_index = \
            process_index

      if recomp:
        cfg.allow_recompute = True
      cfg.optimizations.merge_remote_buffers = merge_remote_buffers
      cfg.optimizations.minimum_remote_tensor_size = minimum_remote_tensor_size
      if tu.has_ci_ipus():
        tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()
      report_json = tu.ReportJSON(test_wrapper, session)

      session.run(variables.global_variables_initializer())
      session.run(infeed_queue.initializer)
      report_json.reset()
      session.run(compiled_model_pipeline,
                  feed_dict=dict(zip(inputs, input_values)))
      out = session.run(outfeed_op)
      if profiling:
        report_json.parse_log()
        if not device_mapping:
          device_mapping = [
              i - (i % 4) + ((i % 4) if (i % 4) < 2 else 5 - (i % 4))
              for i in range(len(stages))
          ]
        report_json.assert_pipeline_stages_on_expected_ipu(
            device_mapping, cfg.ipu_model.tiles_per_ipu)

        report = pva.openReport(report_helper.find_report())
        test_wrapper.assert_max_tile_memory(report,
                                            expected_max_tile_memory,
                                            tolerance=0.5)
      out = out[0] if optimizer else out
      if return_report:
        return out, report_json, report_helper
      elif return_vars:
        return out, _get_vars(session, "ipu")
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
                              batch_serialization_iterations=1,
                              recomputation_mode=None,
                              number_of_io_tiles=0,
                              return_report=False):

    if batch_serialization_iterations > 1:
      assert device_mapping is None
      device_mapping = [0] * len(stages)

    # Run pipeline_on_ipu before the CPU version as pipeline_on_ipu
    # will initialize the IPU which might be needed by the CPU path
    # to run some instructions which only exist on the IPU.
    pipeline_losses = PipelineTester.pipeline_on_ipu(
        stages,
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
        device_mapping,
        batch_serialization_iterations,
        recomputation_mode,
        number_of_io_tiles=number_of_io_tiles,
        return_report=return_report)

    if return_report:
      pipeline_losses, report_json, report_helper = pipeline_losses

    num_batches_to_accumulate = (gradient_accumulation_count *
                                 batch_serialization_iterations)
    cpu_losses = PipelineTester._cpu_with_grad_accum(
        test_wrapper, stages, inputs_fn, input_values, repeat_count,
        num_batches_to_accumulate, dataset_fn, optimizer)

    test_wrapper.assertAllClose(cpu_losses, pipeline_losses)

    if return_report:
      return report_json, report_helper
    return None

  @staticmethod
  def compare_pipeline_to_sharding(
      stages,
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
      batch_serialization_iterations=1,
      recomputation_mode=None,
      number_of_io_tiles=0,
      offload_activations=None,
      merge_remote_buffers=MergeRemoteBuffersBehaviour.IF_BENEFICIAL,
      replication_factor=1,
      replicated_optimizer_state_sharding=False,
      minimum_remote_tensor_size=128):
    if batch_serialization_iterations > 1:
      assert device_mapping is None
      device_mapping = [0] * len(stages)

    if not isinstance(replicated_optimizer_state_sharding, (tuple, list)):
      replicated_optimizer_state_sharding = [
          replicated_optimizer_state_sharding,
          replicated_optimizer_state_sharding
      ]

    pipeline_losses, pipeline_vars = PipelineTester.pipeline_on_ipu(
        stages,
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
        device_mapping,
        batch_serialization_iterations,
        recomputation_mode,
        number_of_io_tiles,
        False,
        replication_factor,
        offload_activations,
        merge_remote_buffers,
        replicated_optimizer_state_sharding[0],
        minimum_remote_tensor_size,
        return_vars=True)

    num_batches_to_accumulate = (gradient_accumulation_count *
                                 batch_serialization_iterations)
    sharded_losses, sharded_vars = PipelineTester._sharded_on_ipu(
        stages,
        inputs_fn,
        input_values,
        repeat_count,
        num_batches_to_accumulate,
        dataset_fn,
        optimizer,
        test_wrapper,
        recomp,
        device_mapping,
        number_of_io_tiles,
        replication_factor,
        merge_remote_buffers,
        replicated_optimizer_state_sharding[1],
        minimum_remote_tensor_size,
        return_vars=True)
    test_wrapper.assertAllClose(sharded_losses, pipeline_losses)
    test_wrapper.assertAllClose(sharded_vars, pipeline_vars)

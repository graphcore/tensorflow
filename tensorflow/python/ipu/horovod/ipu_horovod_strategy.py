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
# ==============================================================================
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.ipu import ipu_multi_worker_strategy
from tensorflow.python.ipu.horovod import Sum, Average, size, allreduce, broadcast
from tensorflow.python.training import server_lib


def _to_horovod_op(reduce_op):
  if reduce_op == reduce_util.ReduceOp.SUM:
    return Sum
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return Average

  raise ValueError("Unsupported reduce op: {}".format(reduce_op))


class IPUHorovodStrategyV1(distribute_lib.StrategyV1):
  """This is a distribution strategy using Horovod.

  Usage is very similar to the `IPUMultiWorkerStrategyV1`, with the
  following differences:

  * There is no `cluster_resolver` argument, as Horovod's built-in
    cluster discovery is used. Hence the `TF_CONFIG` environment
    variable containing the cluster configuration is not needed.
  * As Horovod sets up the necessary communication channels,
    starting a `tf.distribute.Server` is not needed either.
  * Launching the cluster should be done with the `mpirun` tool.

  **Example using a custom training loop with pipelining**

  .. code-block:: python

    strategy = IPUHorovodStrategyV1()

    with strategy.scope():

      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(lr, images, labels):
        partial = keras.layers.Dense(256, activation="relu")(images)
        partial = keras.layers.Dense(128, activation="relu")(partial)
        return lr, partial, labels

      def stage2(lr, partial, labels):
        logits = keras.layers.Dense(10)(partial)
        per_example_loss = keras.losses.sparse_categorical_crossentropy(
            y_true=labels, y_pred=logits, from_logits=True)
        # In a custom training loop, the optimiser does an allreduce *sum*, not
        # average, of the gradients across the distributed workers. Therefore
        # we want to divide the loss here by the *global* batch size, which is
        # done by the `tf.nn.compute_average_loss()` function.
        loss = nn.compute_average_loss(per_example_loss)
        return lr, loss

      def optimizer_function(lr, loss):
        optimizer = GradientDescentOptimizer(lr)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

      def model(lr):
        pipeline_op = pipelining_ops.pipeline(
            computational_stages=[stage1, stage2],
            gradient_accumulation_count=gradient_accumulation_count,
            inputs=[lr],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            name="Pipeline")
        return pipeline_op

      def compiled_model(lr):
        with ipu_scope("/device:IPU:0"):
          return ipu_compiler.compile(model, inputs=[lr])

      with ops.device("cpu"):
        lr = array_ops.placeholder(np.float32, [])

      train_op = strategy.run(compiled_model, args=[lr])

      _, per_worker_losses = outfeed_queue.dequeue()

      # Mean across the local `gradient_accumulation_count` batches:
      per_worker_loss = math_ops.reduce_mean(per_worker_losses)

      # Global mean across the distributed workers (since it is already
      # divided by the global batch size above, we do a sum here):
      global_loss = strategy.reduce(ReduceOp.SUM, per_worker_loss)

      config = ipu_utils.create_ipu_config()
      config = ipu_utils.auto_select_ipus(config, num_ipus=2)
      ipu_utils.configure_ipu_system(config)
      ipu_utils.move_variable_initialization_to_cpu()

      with session.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op, {lr: 0.01})
          global_loss_val = sess.run(global_loss)
  """
  _collective_key_base = 0

  def __init__(self, ipu_device="/device:IPU:0", variables_on_host=False):
    # We create an empty cluster here since we will not be using gRPC for communication.
    # All the communication is delegated to Horovod (MPI) below.
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}))

    super().__init__(
        IPUHorovodExtendedV1(self, cluster_resolver, ipu_device,
                             variables_on_host))


class IPUHorovodExtendedV1(ipu_multi_worker_strategy.IPUMultiWorkerExtendedV1):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               variables_on_host):
    super().__init__(container_strategy, cluster_resolver, ipu_device,
                     variables_on_host)
    self._num_workers = size()

  def _reduce_implementation(self, reduce_op, value, destinations, options):
    del destinations
    del options
    return allreduce(value, op=_to_horovod_op(reduce_op))

  def _batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                   options):
    del options
    op = _to_horovod_op(reduce_op)
    return [allreduce(v, op=op) for (v, _) in value_destination_pairs]

  def _broadcast_implementation(self, initial_value, device):
    del device
    return broadcast(initial_value, root_rank=0)


# Export the alias for backwards compability.
IPUHorovodStrategy = IPUHorovodStrategyV1

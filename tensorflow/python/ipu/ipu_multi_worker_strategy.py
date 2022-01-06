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
# ===================================================================
"""
Distributed training
~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import deprecation


class IPUMultiWorkerStrategyV1(distribute_lib.StrategyV1):
  """This is a distribution strategy for synchronous training using
  IPUs on multiple workers with between-graph replication.

  By default variables and ops are placed on the IPU of each worker,
  but variables can optionally be placed on the host by setting
  `variables_on_host=True`. In any case, this strategy will make
  sure that variables are kept in sync between the workers by
  performing multi-worker reductions.

  The multi-worker reductions are done using TensorFlow's
  implementation of collective operations over gRPC.

  **Variable synchronization**

  The default behavior is to sync (allreduce) the variables when
  they are written (sync-on-write). This is a good choice when
  reads are at least as common as writes. However, for variables
  where writes are more common than reads (like metrics or population
  statistics in batch normalization layers), it is beneficial to
  only sync (allreduce) the variables when they are read
  (sync-on-read).

  In both cases, it is important that all the workers participate
  in the sync, otherwise progress will be blocked. Take special care
  in the latter case (with sync-on-read variables), because it implies
  that all the workers need to read these variables at the same time.
  For example, it implies that all the workers must checkpoint the
  model at the same time.

  Sync-on-read variables are placed on the IPU even when variables
  were requested placed on the host (with `variables_on_host=True`),
  because it allows the ops to update the variables directly on the
  IPU without any host involvement. Only when the variable is read,
  it is streamed to the host and allreduced there.

  **Weight updates**

  When used during training with an `Optimizer`, there is an implicit
  allreduce in the `optimizer.apply_gradients()` function (which is
  called from `optimizer.minimize()`). This will automatically cause
  the gradients to be streamed to the host of each worker, allreduced
  between the workers, and then streamed back to the IPU of each worker,
  where identical weight updates are performed (keeping the workers in
  sync). This is done even when the call to `optimizer.apply_gradients()`
  is inside a function passed to `ipu_compiler.compile()`, as the allreduce
  is extracted from the compiled XLA cluster and placed on the host in
  the outside graph (by internally using an
  :func:`~tensorflow.python.ipu.scopes.outside_compilation_scope`).

  When variables are placed on the host, the weight updates should
  also be placed on the host. In other words, the
  `optimizer.compute_gradients()` call should be placed on the IPU,
  while the `optimizer.apply_gradients()` call should be placed
  on the host. This must be done explicitly. In this scenario all
  the "slot" variables used by the optimizer (e.g. the momentum
  accumulator) are then also kept only in host memory and never
  used on the IPU, saving IPU memory.

  **Compatibility**

  `IPUEstimator`: Pass the `IPUMultiWorkerStrategyV1` instance to the
  :class:`~tensorflow.python.ipu.ipu_run_config.RunConfig` as the
  `train_distribute` argument. When variables are placed on the host,
  the `optimizer.apply_gradients()` call should also be placed on the
  host by using the
  :class:`~tensorflow.python.ipu.ipu_estimator.IPUEstimatorSpec`
  `host_call` argument. See full example: :any:`distributed_training`.

  `IPUPipelineEstimator`: Pass the `IPUMultiWorkerStrategyV1` instance to
  the :class:`~tensorflow.python.ipu.ipu_run_config.RunConfig` as the
  `train_distribute` argument. Placing variables on the host is not
  currently supported here.

  Keras `Model.fit`: Not currently supported.

  Custom training loop: Pass the training step function to
  `IPUMultiWorkerStrategyV1.run()`. With variables on
  the IPU, the `optimizer.apply_gradients()` call can be done from
  an XLA compiled IPU function, and the inter-host allreduce will
  be automatically extracted from the compiled XLA cluster and placed
  on the host. With variables on the host, the `optimizer.apply_gradients()`
  call must be explicitly placed on the host.

  **Example using a custom training loop with pipelining**

  .. code-block:: python

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    strategy = IPUMultiWorkerStrategyV1(cluster_resolver)

    sess_config = tf.ConfigProto()
    sess_config = strategy.update_config_proto(sess_config)
    server = tf.distribute.Server(cluster_resolver.cluster_spec(),
                                  job_name=cluster_resolver.task_type,
                                  task_index=cluster_resolver.task_id,
                                  config=sess_config)
    sess_target = server.target

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

      config = ipu.config.IPUConfig()
      config.auto_select_ipus = 2
      config.configure_ipu_system()
      ipu_utils.move_variable_initialization_to_cpu()

      with session_lib.Session(target=sess_target, config=sess_config) as sess:
        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op, {lr: 0.01})
          global_loss_val = sess.run(global_loss)
  """

  _collective_key_base = 0

  @deprecation.deprecated(None, """IPUMultiWorkerStrategy will be deprecated in
    favour of PopDistStrategy""")
  def __init__(self,
               cluster_resolver,
               ipu_device="/device:IPU:0",
               variables_on_host=False):
    super().__init__(
        IPUMultiWorkerExtendedV1(self, cluster_resolver, ipu_device,
                                 variables_on_host))


def _is_inside_compilation():
  graph = ops.get_default_graph()
  attrs = graph._attr_scope_map  # pylint: disable=protected-access

  is_in_xla_context = control_flow_util.GraphOrParentsInXlaContext(graph)
  is_outside_compilation = scopes.OUTSIDE_COMPILATION_NAME in attrs

  return is_in_xla_context and not is_outside_compilation


@tf_contextlib.contextmanager
def _outside_compilation_scope_if_needed(name):
  if _is_inside_compilation():
    with scopes.outside_compilation_scope(name):
      yield
  else:
    yield


def _ipu_device_for_host(ipu_device_string, host_device_string):
  ipu_device = device_lib.DeviceSpec.from_string(ipu_device_string)
  host_device = device_lib.DeviceSpec.from_string(host_device_string)

  # Take distributed info from the host and device info from the IPU.
  ipu_for_host = device_lib.DeviceSpec(job=host_device.job,
                                       replica=host_device.replica,
                                       task=host_device.task,
                                       device_type=ipu_device.device_type,
                                       device_index=ipu_device.device_index)

  return ipu_for_host.to_string()


def _make_identity_op(v):
  name = v.name.replace(":", "_")
  return array_ops.identity(v, name=name)


class IPUDistributedVariable(values.DistributedVariable):  # pylint: disable=abstract-method
  pass


class IPUSyncOnReadVariable(values.SyncOnReadVariable):  # pylint: disable=abstract-method
  pass


class IPUMirroredVariable(values.MirroredVariable):  # pylint: disable=abstract-method
  pass


IPU_VARIABLE_CLASS_MAPPING = {
    "VariableClass": IPUDistributedVariable,
    variable_scope.VariableSynchronization.AUTO: IPUMirroredVariable,
    variable_scope.VariableSynchronization.ON_WRITE: IPUMirroredVariable,
    variable_scope.VariableSynchronization.ON_READ: IPUSyncOnReadVariable,
}


class IPUAutoPolicy(values.AutoPolicy):  # pylint: disable=abstract-method
  pass


class IPUOnWritePolicy(values.OnWritePolicy):  # pylint: disable=abstract-method
  pass


class IPUOnReadPolicy(values.OnReadPolicy):  # pylint: disable=abstract-method
  pass


IPU_VARIABLE_POLICY_MAPPING = {
    variable_scope.VariableSynchronization.AUTO: IPUAutoPolicy,
    variable_scope.VariableSynchronization.ON_WRITE: IPUOnWritePolicy,
    variable_scope.VariableSynchronization.ON_READ: IPUOnReadPolicy,
}


class IPUMultiWorkerExtendedV1(
    collective_all_reduce_strategy.CollectiveAllReduceExtended):
  def __init__(self, container_strategy, cluster_resolver, ipu_device,
               variables_on_host):
    communication_options = collective_util.Options(
        implementation=cross_device_ops_lib.CollectiveCommunication.RING)
    super().__init__(container_strategy,
                     cluster_resolver=cluster_resolver,
                     communication_options=communication_options)

    host_devices = self._devices
    if len(host_devices) != 1:
      raise ValueError("Expected one host device per worker")

    self._host_device = host_devices[0]
    self._ipu_device = _ipu_device_for_host(ipu_device, self._host_device)
    self._variables_on_host = variables_on_host

    if variables_on_host:
      self._variable_device = self._host_device
    else:
      self._variable_device = self._ipu_device

    # By default the functional graphs are not retraced and therefore device
    # information is not lowered to ops which means distribution strategies do
    # not work.
    self._retrace_functions_for_each_device = True

  def _get_variable_creator_initial_value(self, replica_id, device,
                                          primary_var, **kwargs):
    assert replica_id == 0
    assert device is not None
    assert primary_var is None

    def initial_value_fn():  # pylint: disable=g-missing-docstring
      # Override colocation and XLA attributes for initializers.
      colocation_list = attr_value_pb2.AttrValue.ListValue(s=[b'loc:@cpu'])
      attrs = {
          "_class": attr_value_pb2.AttrValue(list=colocation_list),
          "_XlaCompile": attr_value_pb2.AttrValue(b=False),
          "_XlaScope": attr_value_pb2.AttrValue(s=b''),
      }
      with ops.device(device), \
          ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
        initial_value = kwargs["initial_value"]
        if callable(initial_value):
          initial_value = initial_value()
        assert not callable(initial_value)
        initial_value = ops.convert_to_tensor(initial_value,
                                              dtype=kwargs.get("dtype", None))
        return self._broadcast_implementation(initial_value, device)

    return initial_value_fn

  def _create_variable(self, next_creator, **kwargs):
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      devices = [self._variable_device]
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(**kwargs)
    else:
      devices = colocate_with._devices  # pylint: disable=protected-access

    def _real_creator(**kwargs):
      assert len(devices) == 1
      assert devices[0] == self._variable_device

      # The chief worker will initialize and broadcast the value to
      # the other workers. Always done on the host.
      kwargs["initial_value"] = self._get_variable_creator_initial_value(
          replica_id=0,  # First (and only) replica on each worker.
          device=self._host_device,
          primary_var=None,
          **kwargs)

      # We always place sync-on-read variables on the IPU. They will
      # be transfered and reduced on the hosts only when read.
      synchronization = kwargs.get("synchronization")
      if (not self._variables_on_host or
          synchronization == variable_scope.VariableSynchronization.ON_READ):
        with ops.device(self._ipu_device):
          return [next_creator(**kwargs)]

      # Cache a snapshot of the variable on the IPU device,
      # otherwise the XLA cluster containing the ops consuming the
      # variable might be moved to the host to be colocated with it.
      kwargs["caching_device"] = self._ipu_device

      # In case we are inside an ipu_jit_scope, we need to override it
      # to disable XLA for variable initialization on the host.
      disable_xla = {
          "_XlaCompile": attr_value_pb2.AttrValue(b=False),
          "_XlaScope": attr_value_pb2.AttrValue(s=b''),
      }

      graph = ops.get_default_graph()
      with ops.device(self._host_device), \
          graph._attr_scope(disable_xla):  # pylint: disable=protected-access
        return [next_creator(**kwargs)]

    return distribute_utils.create_mirrored_variable(
        self._container_strategy(), _real_creator, IPU_VARIABLE_CLASS_MAPPING,
        IPU_VARIABLE_POLICY_MAPPING, **kwargs)

  def read_var(self, var):
    return var.read_value()

  def _reduce_to(self, reduce_op, value, destinations, options):
    if isinstance(value, values.DistributedValues):
      assert len(value.values) == 1
      value = value.values[0]

    # Make sure the reduction is done on the host device by wrapping the inputs
    # in an identity op before and after placing it on that device. This also
    # disables the scoped_allocator_optimizer because it allows it to see that
    # we cross a device boundary here.
    value = _make_identity_op(value)
    with _outside_compilation_scope_if_needed("host_reduce"):
      with ops.device(self._host_device):
        value = _make_identity_op(value)

      return self._reduce_implementation(reduce_op, value, destinations,
                                         options)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
    # Make sure the reduction is done on the host device by wrapping the inputs
    # in an identity op before and after placing it on that device. This also
    # disables the scoped_allocator_optimizer because it allows it to see that
    # we cross a device boundary here.
    value_destination_pairs = [(_make_identity_op(v), d)
                               for (v, d) in value_destination_pairs]
    with _outside_compilation_scope_if_needed("host_batch_reduce"):
      with ops.device(self._host_device):
        value_destination_pairs = [(_make_identity_op(v), d)
                                   for (v, d) in value_destination_pairs]

      return self._batch_reduce_implementation(reduce_op,
                                               value_destination_pairs,
                                               options)

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(self._container_strategy(),
                                       replica_id_in_sync_group=0), ops.device(
                                           self._ipu_device):
      # Make sure it is compiled as a single engine when called in graph mode.
      # This is similar to the mechanism used by xla.compile.
      xla_context = control_flow_ops.XLAControlFlowContext()
      try:
        xla_context.Enter()
        outputs = fn(*args, **kwargs)
      finally:
        xla_context.Exit()

      return outputs

  def _validate_colocate_with_variable(self, colocate_with_variable):
    if colocate_with_variable.device != self._variable_device:
      raise ValueError("Unexpected colocated variable device: {}".format(
          colocate_with_variable.device))

  def _reduce_implementation(self, reduce_op, value, destinations, options):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._reduce_to(reduce_op, value, destinations, options)

  def _batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                   options):
    # This is an extension point for overriding, try to keep a stable API.
    return super()._batch_reduce_to(reduce_op, value_destination_pairs,
                                    options)

  def _broadcast_implementation(self, initial_value, device):
    # This is an extension point for overriding, try to keep a stable API.

    if self._num_workers <= 1:
      return initial_value

    assert device is not None
    # Only the first device participates in the broadcast of initial values.
    group_key = self._collective_keys.get_group_key([device])
    group_size = self._num_workers
    collective_instance_key = (self._collective_keys.get_instance_key(
        group_key, device))

    if self._is_chief:
      bcast_send = collective_ops.broadcast_send(initial_value,
                                                 initial_value.shape,
                                                 initial_value.dtype,
                                                 group_size, group_key,
                                                 collective_instance_key)
      with ops.control_dependencies([bcast_send]):
        return array_ops.identity(initial_value)
    else:
      return collective_ops.broadcast_recv(initial_value.shape,
                                           initial_value.dtype, group_size,
                                           group_key, collective_instance_key)


# Export the alias for backwards compability.
IPUMultiWorkerStrategy = IPUMultiWorkerStrategyV1

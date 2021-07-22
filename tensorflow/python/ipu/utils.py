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
"""
General utilities
~~~~~~~~~~~~~~~~~
"""
import os
import time
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
# Adds the enum SyntheticDataCategory into the scope so it can be imported from
# this file. It is required for calling use_synthetic_data_for.
from tensorflow.compiler.plugin.poplar.driver.config_pb2 import SyntheticDataCategory  # pylint: disable=W0611
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
# pylint: disable=unused-import
# These imports are only here to make it easier for the Tensorflow Wheel users
# to use these functions:
# ```
# from tensorflow.python import ipu
# ...
# ipu.utils.export_variables_from_live_session(...)
# ```
from tensorflow.compiler.plugin.poplar.tools.tensorflow_weights_extractor import (
    export_variables_from_live_session, export_variables_from_live_model,
    import_data_in_live_session, import_data_in_live_model)
from tensorflow.compat.v1 import executing_eagerly
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import values
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import dataset_extractor
from tensorflow.python.ipu.config import IPUConfig, SelectionOrder, ExecutionProfileType, DeviceConnectionType, MergeRemoteBuffersBehaviour, SchedulingAlgorithm, KeyId, VerificationOptions, get_ipu_config, configure_ipu_system
# pylint: enable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation


def get_num_of_ipus_in_device(ipu_device, device="cpu"):
  """Get the number of physical IPUs

  Args:
    ipu_device: The IPU device for which to get the number of devices for.
    device: The CPU device which is local to the IPU hardware.

  Returns:
    A number of physical IPUs configured for a particular TF device.
  """

  g = ops.Graph()
  with g.as_default():
    with ops.device(device):
      cfg_op = gen_ipu_ops.ipu_get_num_devices(ipu_device)

  with session_lib.Session(graph=g) as sess:
    return sess.run(cfg_op)


def running_on_ipu_model():
  """ Check if XLA is configured to run on the ipu model.

  Returns:
    True if XLA is configured to run on the ipu model.
    False if XLA is configured to run on real hardware.
  """
  return "--use_ipu_model" in os.environ.get("TF_POPLAR_FLAGS", "")


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def create_ipu_config(profiling=False,
                      enable_ipu_events=False,
                      use_poplar_text_report=False,
                      use_poplar_cbor_report=False,
                      profile_execution=None,
                      enable_poplar_serialized_graph=False,
                      report_every_nth_execution=0,
                      max_report_size=0x10000000,
                      report_directory="",
                      scheduler_selection=SchedulingAlgorithm.CHOOSE_BEST,
                      always_rearrange_copies_on_the_host=False,
                      merge_infeed_io_copies=False,
                      disable_graph_outlining=False,
                      retain_control_dependencies=False,
                      max_cross_replica_sum_buffer_size=0,
                      max_inter_ipu_copies_buffer_size=0,
                      max_scheduler_lookahead_depth=5,
                      max_scheduler_search_space_size=64,
                      prefetch_data_streams=True,
                      selection_order=SelectionOrder.AUTO,
                      enable_experimental_remote_buffer_embedding=False):
  """Create an empty IPU session configuration structure.

  Args:
    profiling: Enable compilation reports, and IPU trace events.
    enable_ipu_events: Enable IPU trace events without Poplar reports.
    use_poplar_text_report: Enable the Poplar textual report summary.
    use_poplar_cbor_report: Enable the Poplar CBOR reports.
    profile_execution: Include Poplar execution profiles in the execution
      events. Can only be enabled if `profiling` is also enabled. If set, can be
      `True`, 'False`, or a member of the `ExecutionProfileType` enumeration.
      A `True` value indicates `ExecutionProfileType.DEVICE_PROFILE`.
    enable_poplar_serialized_graph: Create the Poplar serialized graph and
      include in the IPU compilation trace events.
    report_every_nth_execution: Only produce an execution report on every Nth
      execution.  0 = One report only.
    max_report_size: The maximum size of Poplar profiles to include in the
      profile events.
    report_directory: When set, reports will be written to files in this
      directory, instead of being written into the events.  The events will
      contain the full paths of the report files.
    scheduler_selection: A `SchedulingAlgorithm`. By default, several schedules
      will be created and the one with the lowest predicted liveness chosen.
      Setting this to a specific scheduling algorithm forces the compiler to use
      that algorithm when ordering the instructions.
    always_rearrange_copies_on_the_host: *** Experimental Flag ***
      The data which is streamed to/from the device might be stored in different
      layouts on the device and on the host. If that is the case the
      rearrangement is performed on the device by default. By enabling this
      option the rearrangement will be performed on the host at the expense of
      latency.
    merge_infeed_io_copies: When true, this flag will merge the streamed
      host->device input copies into one larger copy.  This may reduce the time
      to copy data from the host, at the expense of increasing the live tensor
      memory on the device.
    disable_graph_outlining: By default, some operations, such as matrix
      multiplications, which occur in the graph multiple times but with
      different input tensors might be optimised to reduce the total code size
      of the graph at the expense of the execution time. Setting this flag will
      disable these optimisations.
    retain_control_dependencies: When set to true, control dependencies from the
      Tensorflow graph are passed through to the backend.  This can result in a
      different memory size due to differing constraints on the operation
      scheduler.
    max_cross_replica_sum_buffer_size: The maximum number of bytes that can be
      waiting before a cross replica sum op is scheduled.
    max_inter_ipu_copies_buffer_size: The maximum number of bytes that can be
      waiting before a inter IPU copy between IPUs is scheduled.
    max_scheduler_lookahead_depth: The maximum distance to look into the future
      when considering valid schedules.
    max_scheduler_search_space_size: The maximum number of nodes to consider
      when building the tree of future schedules.
    prefetch_data_streams: When set to true, the prefetching of data for data
      streams on the host will be overlapped with execution on the IPU.
    selection_order: the order in which IPUs are selected and mapped to physical
      IPU devices when using a multi-IPU devices (see `SelectionOrder`). By
      default, automatic selection order is used.
    enable_experimental_remote_buffer_embedding: When set to true,
      `HostEmbedding` will make use of Poplar remote buffers.

  Returns:
    An IpuOptions configuration protobuf, suitable for passing to
    ``configure_ipu_system``
  """
  if profiling and enable_ipu_events:
    raise Exception(
        "`profiling` and `enable_ipu_events` are mutually exclusive")

  profile_execution = profile_execution if profile_execution \
                                        else ExecutionProfileType.NO_PROFILE

  if isinstance(profile_execution, (np.bool_, bool)):
    if profile_execution:
      profile_execution = ExecutionProfileType.DEVICE_PROFILE
    else:
      profile_execution = ExecutionProfileType.NO_PROFILE

  if (profile_execution != ExecutionProfileType.NO_PROFILE and not profiling):
    raise Exception("`profiling` is required when `profile_execution` is set")

  if not isinstance(profile_execution, ExecutionProfileType):
    raise Exception("`profile_execution` must be True, False, or an "
                    "ExecutionProfileType instance")

  opts = config_pb2.IpuOptions()

  # Default initialize IpuOptions() attributes here.
  opts.creator_id = config_pb2.IpuOptionsCreator.IPU_UTILS
  opts.ipu_model_config.compile_ipu_code = True
  opts.ipu_model_config.ipu_model_version = "ipu2"
  opts.enable_multi_slice_combiner = False
  opts.enable_matmul_combiner = False
  opts.disable_gather_simplifier = False
  opts.device_connection_type = DeviceConnectionType.ALWAYS.value
  opts.speed_size_config.allow_recompute = False
  opts.remote_buffer_merging_mode = threestate_pb2.THREESTATE_UNDEFINED

  # Configure IpuOptions according to the passed arguments.
  opts.profiling.enable_ipu_trace_events = profiling or enable_ipu_events
  opts.profiling.enable_compilation_trace = profiling or enable_ipu_events
  opts.profiling.enable_io_trace = profiling or enable_ipu_events
  opts.profiling.execution_trace_type = profile_execution.value
  opts.profiling.enable_poplar_reports_text = use_poplar_text_report
  opts.profiling.enable_poplar_reports_cbor = use_poplar_cbor_report
  opts.profiling.enable_poplar_graph = enable_poplar_serialized_graph
  opts.profiling.report_every_nth_execution = report_every_nth_execution
  opts.profiling.max_report_size = max_report_size
  opts.profiling.report_directory = report_directory

  opts.speed_size_config.always_rearrange_copies_on_the_host = \
      always_rearrange_copies_on_the_host
  opts.speed_size_config.merge_infeed_io_copies = merge_infeed_io_copies
  opts.speed_size_config.disable_graph_outlining = \
      disable_graph_outlining
  if isinstance(scheduler_selection, str):
    deprecation_mapping = {
        "": SchedulingAlgorithm.CHOOSE_BEST,
        "Clustering": SchedulingAlgorithm.CLUSTERING,
        "PostOrder": SchedulingAlgorithm.POST_ORDER,
        "LookAhead": SchedulingAlgorithm.LOOK_AHEAD,
        "ShortestPath": SchedulingAlgorithm.SHORTEST_PATH
    }
    if scheduler_selection not in deprecation_mapping:
      raise TypeError(f"Could not convert '{scheduler_selection}' to a valid"
                      " `SchedulingAlgorithm`.")
    scheduler_selection = deprecation_mapping[scheduler_selection]
  opts.speed_size_config.scheduler_selection = scheduler_selection.value

  opts.retain_control_dependencies = retain_control_dependencies
  opts.max_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size
  opts.max_inter_ipu_copies_buffer_size = max_inter_ipu_copies_buffer_size
  opts.minimum_remote_tensor_size = 128

  opts.max_scheduler_lookahead_depth = max_scheduler_lookahead_depth
  opts.max_scheduler_search_space_size = max_scheduler_search_space_size

  opts.prefetch_data_streams = prefetch_data_streams
  opts.selection_order = selection_order.value

  opts.verified_transfers.enabled = False
  opts = set_verification_options(opts, VerificationOptions())

  opts.enable_experimental_remote_buffer_embedding = \
      enable_experimental_remote_buffer_embedding

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_serialization_options(opts, output_folder=""):
  """ Enable / disable the serialization to disk of the compiled executables.

  .. code-block:: python

      # Create a device that will save to disk all the compiled executables.
      opts = create_ipu_config()
      opts = set_serialization_options(opts,
                                      output_folder="/tmp/my_network")
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    output_folder: Where to save the compiled executables.
                   Set to "" to disable serialization.

  Returns:
    The IpuOptions configuration protobuf.
  """
  opts.serialization_folder = output_folder
  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_optimization_options(
    opts,
    combine_embedding_lookups=False,
    combine_matmuls=False,
    max_cross_replica_sum_buffer_size=0,
    max_reduce_scatter_buffer_size=0,
    max_inter_ipu_copies_buffer_size=0,
    max_send_recv_cluster_size=0,
    minimum_remote_tensor_size=128,
    merge_remote_buffers=MergeRemoteBuffersBehaviour.NO_MERGING,
    gather_simplifier=True,
    triangular_solve_expander_block_size=0,
    cholesky_block_size=0,
    enable_fast_math=False):
  """Set the IPU options related to performance / optimizations.

  .. code-block:: python

      # Create a device with fusion for multiSlices sharing the same input
      # enabled.
      opts = create_ipu_config()
      opts = set_optimization_options(opts,
                                      combine_embedding_lookups=True)
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    combine_embedding_lookups: Fuse embedding lookups on the same tensor. This
      might improve performance but increase memory usage.
    combine_matmuls: Fuse matmul operations if they share the same weights or
      the same input.
    max_cross_replica_sum_buffer_size: The maximum number of bytes that can be
      waiting before a cross replica sum op is scheduled.
    max_reduce_scatter_buffer_size: The maximum number of bytes that can be
      waiting before a reduce scatter op is scheduled.
    max_inter_ipu_copies_buffer_size: The maximum number of bytes that can be
      waiting before a inter IPU copy between IPUs is scheduled.
    max_send_recv_cluster_size: The maximum number of bytes that can be waiting
      before a cluster of send/recv instructions to/from the host is scheduled.
      These are lowered to stream copies that can be merged by Poplar.
    minimum_remote_tensor_size: The minimum size (in bytes) a tensor has to be
      in order to be consider for being stored in remote memory.
    merge_remote_buffers: Whether to merge compatible remote buffers. Merging
      of remote buffers can allow for more code re-use if the only difference
      between computations are the remote buffers being accessed. Must be a
      :py:class:`~tensorflow.python.ipu.config.MergeRemoteBuffersBehaviour`.
      Defaults to `MergeRemoteBuffersBehaviour.NO_MERGING`.
    gather_simplifier: Will enable more aggressive optimisations for embedding
      lookups.
    triangular_solve_expander_block_size: Defines size for triangular solver
      expander blocks. The processing within each block is performed on a
      single tile. The control code for performing computations over blocks
      are unrolled on the device. For a matrix of rank ``N`` and block size
      ``B``, there are ``log2(N/B)`` iterations of the control code. The choice
      of this parameter therefore has to balance between the amount of data in
      a tile (lower value is better, gives better parallelism) and the amount
      of control code (larger value is better, less control code). A value of 0
      selects an implementation defined default.
    cholesky_block_size: Defines the block size for the Cholesky factoriser.
      The processing within each block is performed on a single tile. The
      control code for performing computations over blocks are unrolled on the
      device. For a matrix of rank ``N`` and block size ``B``, there are
      ``N/B`` iterations of the control code. The choice of this parameter
      therefore has to balance between the amount of data in a tile (lower
      value is better, gives better parallelism) and the amount of control code
      (larger value is better, less control code). A value of 0 selects an
      implementation defined default.
    enable_fast_math: Enables optimizations which allow arbitrary reassociations
      and transformations of mathematical operations with no accuracy
      guarantees. Enabling this option can result in incorrect output for
      programs that depend on an exact implementation of IEEE for math
      functions. It may, however, yield faster code for programs that do not
      require the guarantees of these specifications.

  Returns:
    The IpuOptions configuration protobuf.
  """
  def bool_to_three_state(value):
    if value is None:
      return threestate_pb2.THREESTATE_UNDEFINED
    elif value:
      return threestate_pb2.THREESTATE_ON
    return threestate_pb2.THREESTATE_OFF

  # Backwards compatibility
  if not isinstance(merge_remote_buffers, MergeRemoteBuffersBehaviour):
    merge_remote_buffers = bool_to_three_state(merge_remote_buffers)
  else:
    merge_remote_buffers = merge_remote_buffers.value

  # Internally embedding lookups are implemented using multiSlice operations.
  opts.enable_multi_slice_combiner = combine_embedding_lookups
  opts.enable_matmul_combiner = combine_matmuls
  opts.max_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size
  opts.max_reduce_scatter_buffer_size = max_reduce_scatter_buffer_size
  opts.max_inter_ipu_copies_buffer_size = max_inter_ipu_copies_buffer_size
  opts.max_send_recv_cluster_size = max_send_recv_cluster_size
  opts.minimum_remote_tensor_size = minimum_remote_tensor_size
  opts.remote_buffer_merging_mode = merge_remote_buffers
  opts.disable_gather_simplifier = not gather_simplifier
  opts.triangular_solve_expander_block_size = \
    triangular_solve_expander_block_size
  opts.cholesky_block_size = cholesky_block_size
  opts.enable_fast_math = enable_fast_math

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_norm_options(opts,
                     use_stable_statistics=False,
                     experimental_distributed_batch_norm_replica_group_size=1):
  """Set the IPU options related to normalisation operations.
  Note that these
  options will be applied to all normalisation operations encountered
  (Fused Batch Norm, IPU Specific Group Norm, IPU Specific Layer Norm and IPU
  Specific Instance Norm).

  Args:
    use_stable_statistics: If True, computes the mean first and subtracts
      the activations by it before computing the variance. The
      implementation with this flag set to True is slower than when set
      to False.
    experimental_distributed_batch_norm_replica_group_size: When executing
      training fused batch norm operations, this option specifies how many
      replicas to aggregate the batch statistics across.
      For example if a model is being executed across four replicas and this
      option is set to two, replicas 0 and 1 will be grouped together and
      replicas 2 and 3 will be grouped together and the batch norm statistics
      will be synchronously all-reduced every time the layer is executed across
      the replicas within a group.
      This option should not be used when using model parallelism (pipelining)
      across multiple IPUs and it is not supported with I/O tiles.
      When recomputation is enabled and the training fused batch norm operation
      is recomputed, the statistics will have to be all-reduced again, unless
      the
      `pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved`
      recomputation mode is used.
      This option is experimental and may be removed with short/no notice.

  Returns:
    The IpuOptions configuration protobuf.
  """
  if experimental_distributed_batch_norm_replica_group_size < 1:
    raise ValueError("experimental_distributed_batch_norm_replica_group_size"
                     "needs to be at least 1.")
  opts.use_stable_norm_statistics = use_stable_statistics
  opts.experimental_distributed_batch_norm_replica_group_size = \
    experimental_distributed_batch_norm_replica_group_size

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_transfer_options(opts, use_verified_transfers=False):
  """Set the IPU options related to Poplar data transfers.

  Args:
    opts: An IpuOptions session control protobuf.
    use_verified_transfers: If True, use Poplar's verified transfers.

  Returns:
    The IpuOptions configuration protobuf.
  """
  opts.verified_transfers.enabled = use_verified_transfers

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_verification_options(opts, verification_options):
  """Configure verified transfers.
     Set the pairs or key / id to use for each type of data used in the graph
     when verified transfers are enabled.

  .. code-block:: python

      # Create a device which will use verified transfers with different keys.
      opts = create_ipu_config()
      opts = set_transfer_options(opts, use_verified_transfers=True)
      o = VerificationOptions()
      o.input_parameters = KeyId(1)
      o.infeeds["training_feed"] = KeyId(2)
      opts = set_verification_options(opts, o)
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    verification_options: a VerificationOptions object that contains
      the keys / ids to use.
  """
  if not isinstance(verification_options, VerificationOptions):
    raise Exception(
        "`verification_options` must be of type VerificationOptions")

  def _cp_key_and_id(src, dst):
    dst.key = src.key
    dst.start_id = src.start_id

  for attr in [
      "inputs", "input_parameters", "outputs", "output_parameters",
      "checkpoint_in", "checkpoint_out"
  ]:
    _cp_key_and_id(getattr(verification_options, attr),
                   getattr(opts.verified_transfers, attr))

  for name, options in verification_options.infeeds.items():
    _cp_key_and_id(options, opts.verified_transfers.infeeds[name])

  for name, options in verification_options.outfeeds.items():
    _cp_key_and_id(options, opts.verified_transfers.outfeeds[name])

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_compilation_options(opts, compilation_options=None):
  """Set the IPU compilation options for the session.

  .. code-block:: python

      # Create a device with debug execution profile flag set to "compute_sets"
      opts = create_ipu_config()
      opts = set_compilation_options(opts,
          compilation_options={"debug.instrument": "true",
                               "debug.allowOutOfMemory": "true"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    compilation_options: A dictionary of Poplar compilation option flags to be
      sent to the executor.

  Returns:
    The IpuOptions configuration protobuf, with engine compilation options set.
  """
  if compilation_options:
    if not isinstance(compilation_options, dict):
      raise Exception("`compilation_options` must be a dictionary")

    for (option_name, value) in compilation_options.items():
      compilation_option = opts.compilation_options.add()
      compilation_option.option = option_name
      compilation_option.value = value

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_convolution_options(opts, convolution_options=None):
  """Set the IPU convolution options for the session.

  .. code-block:: python

      # Set "availableMemoryProportion" flag to "0.1"
      opts = create_ipu_config()
      opts = set_convolution_options(opts,
          convolution_options={"availableMemoryProportion": "0.1"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    convolution_options: A dictionary of Poplar option flags for
      convolutions. The "availableMemoryProportion" flag indicates the
      proportion of tile memory to be made available as
      temporary memory for convolutions (float between 0 and 1.0).
      Less temporary memory will generally result in a convolution that
      takes more cycles to complete. However, because always live memory
      (such as control code and vertex state) is not tracked when planning it,
      a convolution using less temporary memory may use more memory overall,
      due to an increase of always live memory.

  Returns:
    The IpuOptions configuration protobuf, with convolution options set.
  """
  if convolution_options:
    if not isinstance(convolution_options, dict):
      raise Exception("`convolution_options` must be a dictionary")

    for (option_name, value) in convolution_options.items():
      opt = opts.convolution_options.add()
      opt.option = option_name
      opt.value = value

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_matmul_options(opts, matmul_options=None, clear_pass_type=False):
  """Set the IPU matrix multiplication options for the session.

  .. code-block:: python

      # Set "availableMemoryProportion" flag to "0.5"
      opts = create_ipu_config()
      opts = set_matmul_options(opts,
          matmul_options={"availableMemoryProportion": "0.5"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    matmul_options: A dictionary containing the Poplar option flag
      "availableMemoryProportion" for the matrix multiplication operations.
      It indicates the proportion of tile memory to be made available as
      temporary memory for the matrix multiplications (float between 0 and 1.0).
      Less temporary memory will generally result in a multiplication that
      takes more cycles to complete. However, because always live memory
      (like code and vertex state) is not tracked when planning it,
      a multiplication using less temporary memory may use more memory overall,
      due to an increase of always live memory.
    clear_pass_type: When set to True, the Pass type will not
      be set in the options passed to the Poplar operation.

  Returns:
    The IpuOptions configuration protobuf, with matmul options set.
  """
  if matmul_options:
    if not isinstance(matmul_options, dict):
      raise Exception("`matmul_options` must be a dictionary")

    for (option_name, value) in matmul_options.items():
      opt = opts.matmul_options.add()
      opt.option = option_name
      opt.value = value

  opts.clear_matmul_pass_type = clear_pass_type

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_pooling_options(opts, pooling_options=None):
  """Set the IPU pooling compilation options for the session.

  .. code-block:: python

      # Set "poolUseIntrospectiveMapping" flag to "false"
      opts = create_ipu_config()
      opts = set_pooling_options(opts,
          pooling_options={"poolUseIntrospectiveMapping": "false"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    pooling_options: A dictionary of Poplar option flags for the pooling
      operation.

  Returns:
    The IpuOptions configuration protobuf, with pooling options set.
  """
  if pooling_options:
    if not isinstance(pooling_options, dict):
      raise Exception("`pooling_options` must be a dictionary")

    for (option_name, value) in pooling_options.items():
      opt = opts.pooling_options.add()
      opt.option = option_name
      opt.value = value

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_report_options(opts, graph_options=None, execution_options=None):
  """Set the options used to influence Poplar graph and execution report generation.


  .. code-block:: python

      opts = create_ipu_config()
      opts = set_report_options(opts,
          graph_options={"graphOptions": "false"},
          execution_options={"executionOptions": "false"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    graph_options: A dictionary of Poplar option flags for the graph report
      generation.
    execution_options: A dictionary of Poplar option flags for the execution
      report generation.

  Returns:
    The IpuOptions configuration protobuf, with convolution options set.
  """
  if graph_options:
    if not isinstance(graph_options, dict):
      raise Exception("`graph_options` must be a dictionary")

    for (option_name, value) in graph_options.items():
      opt = opts.profiling.graph_options.add()
      opt.option = option_name
      opt.value = value

  if execution_options:
    if not isinstance(execution_options, dict):
      raise Exception("`execution_options` must be a dictionary")

    for (option_name, value) in execution_options.items():
      opt = opts.profiling.execution_options.add()
      opt.option = option_name
      opt.value = value

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_ipu_model_options(opts,
                          compile_ipu_code=True,
                          tiles_per_ipu=None,
                          ipu_model_version="ipu2"):
  """Set the IPU Model options.

  Args:
    compile_ipu_code: Whether or not to actually compile real IPU code for
      modelling.
    tiles_per_ipu: The number of tiles per IPU Model device.
    ipu_model_version: Specify the IPU version to be used by the IPU Model.
      Must be one of "ipu1" or "ipu2" (default).

  Returns:
    The IpuOptions configuration protobuf, with IPU model options set.
  """
  opts.ipu_model_config.compile_ipu_code = compile_ipu_code
  if tiles_per_ipu:
    opts.ipu_model_config.tiles_per_ipu = tiles_per_ipu
  opts.ipu_model_config.ipu_model_version = ipu_model_version

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_recomputation_options(opts, allow_recompute=True):  # pylint: disable=unused-argument
  """Set re-computation options.

  Args:
    allow_recompute: Whether or not to re-compute instructions during training.
      If this is enabled then we will attempt to pattern match
      instructions/pipeline stages in the forward pass and recompute them in the
      backward pass to avoid having to preserve activations which increase the
      maximum memory liveness. Enabling this option can reduce memory usage at
      the expense of extra computation. Any stateful operations cannot be
      recomputed.

  Returns:
    The IpuOptions configuration protobuf.
  """

  opts.speed_size_config.allow_recompute = allow_recompute

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_floating_point_behaviour_options(opts,
                                         inv=True,
                                         div0=True,
                                         oflo=True,
                                         esr=True,
                                         nanoo=True):
  """Set the IPU floating point control behaviour bits.

  See the Poplar API documentation for poplar::FloatingPointBehaviour.

  Args:
    inv: If true a floating point invalid operation (defined by IEEE 754)
      will cause an exception.
    div0: If true a floating point divide by zero operation will cause an
      exception.
    oflo: If true a floating point overflow will cause an exception.
    esr: Enable stochastic rounding.
    nanoo: Enable Not-a-Number on overflow mode.
  """
  opts.floating_point_behaviour.inv = inv
  opts.floating_point_behaviour.div0 = div0
  opts.floating_point_behaviour.oflo = oflo
  opts.floating_point_behaviour.esr = esr
  opts.floating_point_behaviour.nanoo = nanoo

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_io_tile_options(opts,
                        num_io_tiles,
                        place_ops_on_io_tiles=None,
                        io_tile_available_memory_proportion=0.9):
  """Set the number of tiles reserved for I/O per IPU.

  Args:
    num_io_tiles: Number of tiles to reserve I/O.
    place_ops_on_io_tiles: Whether to place TensorFlow I/O operations on the
      I/O tiles. The value `None` leaves the current value unchanged.
    io_tile_available_memory_proportion: Proportion of I/O tiles memory which
      can be used to store data in, with the remaining memory assumed to be
      used by code. If the size of data which is to be stored on I/O tiles
      exceeds the total I/O tiles memory multiplied by this proportion, then
      a warning message will appear and the operations will not be placed on
      I/O tiles.


  Returns:
    The IpuOptions configuration protobuf.
  """
  opts.num_io_tiles = num_io_tiles
  opts.io_tile_available_memory_proportion = io_tile_available_memory_proportion

  if place_ops_on_io_tiles is not None:
    if place_ops_on_io_tiles and num_io_tiles == 0:
      raise ValueError("Cannot place ops on I/O tiles when num_io_tiles == 0")
    opts.place_ops_on_io_tiles = place_ops_on_io_tiles

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_gcl_options(opts, gcl_options=None):
  """Set the IPU options for the Graphcore Communication Library.

  Args:
    gcl_options: A dictionary with options for configuring the GCL collective
      operations.

  Returns:
    The IpuOptions configuration protobuf.
  """
  if not isinstance(gcl_options, dict):
    raise TypeError("`gcl_options` must be a dictionary")

  for (option_name, value) in gcl_options.items():
    opt = opts.gcl_options.add()
    opt.option = option_name
    opt.value = value

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def auto_select_ipus(opts, num_ipus):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple TensorFlow
  devices, each with control of one of more IPUs. The devices will be labeled
  ``/device:IPU:0``, ``/device:IPU:1`` and so on.

  Each device can control a specific number of IPUs, given by the ``num_ipus``
  parameter. The system will automatically select IPU configurations from the
  available IPUs, where they match the desired number of IPUs.

  Examples:


  .. code-block:: python

    # Create a single device, with one IPU
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=1)
    ipu.utils.configure_ipu_system(opts)
    with tf.Session() as s:
      ...

  .. code-block:: python

    # Create two devices, with 2 IPUs per device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[2,2])
    ipu.utils.configure_ipu_system(opts)
    with tf.Session() as s:
      ...

  .. code-block:: python

    # Create two devices, with 1 IPU in the first device and 2 IPUs
    # in the second device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[1,2])
    ipu.utils.configure_ipu_system(opts)
    with tf.Session() as s:
      ...

  Args:
    opts: An IpuOptions session control protobuf.
    num_ipus: List of IPUs per TensorFlow device

  Returns:
    The IpuOptions configuration protobuf, configured for auto-selecting a set
    of IPU devices.
  """
  if opts.device_config:
    raise Exception("IPU devices have already been configured.")

  if not isinstance(num_ipus, (int, list, tuple)):
    raise Exception("`num_ipus` must be an integer, list or tuple.")

  if isinstance(num_ipus, int):
    dev = opts.device_config.add()
    dev.auto_count = num_ipus
  else:
    for n in num_ipus:
      dev = opts.device_config.add()
      dev.auto_count = n

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def select_ipus(opts, indices):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple TensorFlow
  devices, each with control of one of more IPUs. The TensorFlow devices will be
  labeled ``/device:IPU:0``, ``/device:IPU:1`` and so on.

  Each TensorFlow device uses a specific configuration consisting of one or more
  IPUs from the list of devices.  These can be found by running the Graphcore
  utility ``gc-info -l``.  For instance, the following listing shows the device
  configurations available on a system with 16 IPUs.

  .. code-block:: shell

      user@host:~$ gc-info -l
      Graphcore device listing:

      -+- Id:  [0], type:      [PCIe], PCI Domain: [0000:1a:00.0]
      -+- Id:  [1], type:      [PCIe], PCI Domain: [0000:1b:00.0]
      -+- Id:  [2], type:      [PCIe], PCI Domain: [0000:23:00.0]
      -+- Id:  [3], type:      [PCIe], PCI Domain: [0000:24:00.0]
      -+- Id:  [4], type:      [PCIe], PCI Domain: [0000:3d:00.0]
      -+- Id:  [5], type:      [PCIe], PCI Domain: [0000:3e:00.0]
      -+- Id:  [6], type:      [PCIe], PCI Domain: [0000:43:00.0]
      -+- Id:  [7], type:      [PCIe], PCI Domain: [0000:44:00.0]
      -+- Id:  [8], type:      [PCIe], PCI Domain: [0000:8b:00.0]
      -+- Id:  [9], type:      [PCIe], PCI Domain: [0000:8c:00.0]
      -+- Id: [10], type:      [PCIe], PCI Domain: [0000:8e:00.0]
      -+- Id: [11], type:      [PCIe], PCI Domain: [0000:8f:00.0]
      -+- Id: [12], type:      [PCIe], PCI Domain: [0000:b8:00.0]
      -+- Id: [13], type:      [PCIe], PCI Domain: [0000:b9:00.0]
      -+- Id: [14], type:      [PCIe], PCI Domain: [0000:ba:00.0]
      -+- Id: [15], type:      [PCIe], PCI Domain: [0000:bb:00.0]
      -+- Id: [16], type: [Multi IPU]
      |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
      |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
      -+- Id: [17], type: [Multi IPU]
      |--- PCIe Id:  [4], DNC Id: [0], PCI Domain: [0000:3d:00.0]
      |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:43:00.0]
      -+- Id: [18], type: [Multi IPU]
      |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:24:00.0]
      |--- PCIe Id:  [1], DNC Id: [1], PCI Domain: [0000:1b:00.0]
      -+- Id: [19], type: [Multi IPU]
      |--- PCIe Id:  [2], DNC Id: [0], PCI Domain: [0000:23:00.0]
      |--- PCIe Id:  [0], DNC Id: [1], PCI Domain: [0000:1a:00.0]
      -+- Id: [20], type: [Multi IPU]
      |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
      |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
      -+- Id: [21], type: [Multi IPU]
      |--- PCIe Id: [12], DNC Id: [0], PCI Domain: [0000:b8:00.0]
      |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:ba:00.0]
      -+- Id: [22], type: [Multi IPU]
      |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:8c:00.0]
      |--- PCIe Id: [11], DNC Id: [1], PCI Domain: [0000:8f:00.0]
      -+- Id: [23], type: [Multi IPU]
      |--- PCIe Id: [10], DNC Id: [0], PCI Domain: [0000:8e:00.0]
      |--- PCIe Id:  [8], DNC Id: [1], PCI Domain: [0000:8b:00.0]
      -+- Id: [24], type: [Multi IPU]
      |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
      |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
      |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
      |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
      -+- Id: [25], type: [Multi IPU]
      |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:24:00.0]
      |--- PCIe Id:  [1], DNC Id: [1], PCI Domain: [0000:1b:00.0]
      |--- PCIe Id:  [2], DNC Id: [2], PCI Domain: [0000:23:00.0]
      |--- PCIe Id:  [0], DNC Id: [3], PCI Domain: [0000:1a:00.0]
      -+- Id: [26], type: [Multi IPU]
      |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
      |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
      |--- PCIe Id: [12], DNC Id: [2], PCI Domain: [0000:b8:00.0]
      |--- PCIe Id: [14], DNC Id: [3], PCI Domain: [0000:ba:00.0]
      -+- Id: [27], type: [Multi IPU]
      |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:8c:00.0]
      |--- PCIe Id: [11], DNC Id: [1], PCI Domain: [0000:8f:00.0]
      |--- PCIe Id: [10], DNC Id: [2], PCI Domain: [0000:8e:00.0]
      |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:8b:00.0]
      -+- Id: [28], type: [Multi IPU]
      |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
      |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
      |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
      |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
      |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:24:00.0]
      |--- PCIe Id:  [1], DNC Id: [5], PCI Domain: [0000:1b:00.0]
      |--- PCIe Id:  [2], DNC Id: [6], PCI Domain: [0000:23:00.0]
      |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
      -+- Id: [29], type: [Multi IPU]
      |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
      |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
      |--- PCIe Id: [12], DNC Id: [2], PCI Domain: [0000:b8:00.0]
      |--- PCIe Id: [14], DNC Id: [3], PCI Domain: [0000:ba:00.0]
      |--- PCIe Id:  [9], DNC Id: [4], PCI Domain: [0000:8c:00.0]
      |--- PCIe Id: [11], DNC Id: [5], PCI Domain: [0000:8f:00.0]
      |--- PCIe Id: [10], DNC Id: [6], PCI Domain: [0000:8e:00.0]
      |--- PCIe Id:  [8], DNC Id: [7], PCI Domain: [0000:8b:00.0]
      -+- Id: [30], type: [Multi IPU]
      |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
      |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
      |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
      |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
      |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:24:00.0]
      |--- PCIe Id:  [1], DNC Id: [5], PCI Domain: [0000:1b:00.0]
      |--- PCIe Id:  [2], DNC Id: [6], PCI Domain: [0000:23:00.0]
      |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
      |--- PCIe Id: [13], DNC Id: [8], PCI Domain: [0000:b9:00.0]
      |--- PCIe Id: [15], DNC Id: [9], PCI Domain: [0000:bb:00.0]
      |--- PCIe Id: [12], DNC Id: [10], PCI Domain: [0000:b8:00.0]
      |--- PCIe Id: [14], DNC Id: [11], PCI Domain: [0000:ba:00.0]
      |--- PCIe Id:  [9], DNC Id: [12], PCI Domain: [0000:8c:00.0]
      |--- PCIe Id: [11], DNC Id: [13], PCI Domain: [0000:8f:00.0]
      |--- PCIe Id: [10], DNC Id: [14], PCI Domain: [0000:8e:00.0]
      |--- PCIe Id:  [8], DNC Id: [15], PCI Domain: [0000:8b:00.0]

  Examples based on the listing above:

  .. code-block:: python

      # Create a single device with 1 IPU at PCI address 0000:1a:00.0 by using
      # IPU configuration index 0
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[0])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create a single device with 1 IPU at PCI address 0000:8b:00.0 by using
      # IPU configuration index 8
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[8])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create two TensorFlow devices, with one IPU each, being devices at
      # indices 0 and 1
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[0, 1])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create two TensorFlow devices, with four IPUs each. The device
      # configurations at indices 24 (0000:3e:00.0, 0000:44:00.0, 0000:3d:00.0,
      # 000:43:00.0) and 25 (0000:24:00.0, 0000:1b:00.0, 0000:23:00.0,
      # 00:1a:00.0)
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[24, 25])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create four TensorFlow devices each with one IPU, at addresses
      # 0000:1a:00.0, 0000:1b:00.0, 0000:23:00.0, 0000:24:00.0.
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[0, 1, 2, 3])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    indices: List of IPU configuration indices.
  Returns:
    The IpuOptions configuration protobuf, with a number of devices selected by
    IPU configuration index.
  """

  if opts.device_config:
    raise Exception("IPU devices have already been configured.")

  if not isinstance(indices, (list, tuple)):
    raise Exception("`indices` must be a list or tuple.")

  if len(set(indices)) != len(indices):
    raise Exception("All device indices in `indices` must be unique.")

  for i in indices:
    dev = opts.device_config.add()
    dev.cfg_index = i

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_ipu_connection_type(opts,
                            connection_type=DeviceConnectionType.ALWAYS,
                            ipu_version="",
                            enable_remote_buffers=False):
  """
  Configure when to attach to the device.
  You can use this to, for example,
  compile and cache a program without attaching to an IPU, and then later run
  on a real IPU device without recompiling. Setting the connection type doesn't
  impact the ability to profile a model.

  .. code-block:: python

      # Compile without attaching to the device.
      opts = create_ipu_config()
      opts = set_ipu_connection_type(opts,
                                     DeviceConnectionType.ON_DEMAND))
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    connection_type: One of `DeviceConnectionType`.
                     Defaults to `DeviceConnectionType.ALWAYS`.
    ipu_version: Version of the IPU hardware to use (string). Must be one of
                 "ipu1", "ipu2" or "". Only required if the `connection_type`
                 provided is `DeviceConnectionType.PRE_COMPILE` or
                 `DeviceConnectionType.NEVER`.
    enable_remote_buffers: Default to `False`. When
    :ref:`connection type <device_connection.type>` is
    `DeviceConnectionType.PRE_COMPILE`, `DeviceConnectionType.NEVER` or
    `DeviceConnectionType.ON_DEMAND`, this argument is used to indicate whether
    remote buffers are enabled and supported in the system which will eventually
    be used to execute the compiled programs. Set it to True if the system on
    which you will execute the compiled programs has remote buffers enabled and
    `connection_type` is not `DeviceConnectionType.ALWAYS`. If the
    `connection_type` is `DeviceConnectionType.ALWAYS` then the
    `enable_remote_buffers` parameter is ignored because in that case it is
    possible to query the device.

    In order to check whether your target system supports remote buffers you can
    run the command:

    .. code-block:: console

      $ gc-info -d 0 -I

    If you see ``remote buffers supported: 1`` in the output, that means that
    remote buffers are supported on your system. For more information, see the
    `gc-info documentation
    <https://docs.graphcore.ai/projects/command-line-tools/en/latest/gc-info_main.html>`__.
  Returns:
    The IpuOptions configuration protobuf.
  """

  if ipu_version == "" and connection_type in [
      DeviceConnectionType.NEVER, DeviceConnectionType.PRE_COMPILE
  ]:
    raise Exception("`ipu_version` must be specified when `connection_type` is"
                    f"set to `{connection_type}`.")

  opts.device_connection_type = connection_type.value

  # Passing an int is deprecated
  if isinstance(ipu_version, int):
    ipu_version = "ipu" + str(ipu_version)
  opts.ipu_version = ipu_version

  opts.enable_remote_buffers_without_device = enable_remote_buffers

  return opts


@deprecation.deprecated(
    None, "Configuring IPU session options for TensorFlow has changed and this"
    " function will be removed in a future release. Use an IPUConfig instance"
    " instead. For more information on how to create an equivalent config in"
    " the new IPUConfig API, refer to the API changes for SDK 2.1 in the"
    " TensorFlow documentation.")
def set_experimental_multi_replica_distribution_options(
    opts, process_count, process_index):
  """Configure run-time parallel processes to execute the same Poplar program.
  This will use the Poplar runtime replica subset feature to let multiple
  processes collaborate on executing the same Poplar program by executing a
  subset of the global replicas each.

  The total global replication factor will be equal to the local replication
  factor multiplied by the `process_count`.

  WARNING: This API is experimental and subject to change.

  Args:
    process_count: The total number of processes.
    process_index: The index of the current process.

  Returns:
    The IpuOptions configuration protobuf.
  """

  if not 0 <= process_index < process_count:
    raise ValueError("0 <= process_index < process_count")

  opts.multi_replica_process_count = process_count
  opts.multi_replica_process_index = process_index

  return opts


def reset_ipu_seed(seed,
                   device="/device:IPU:0",
                   cpu_device="cpu",
                   experimental_identical_replicas=False):
  """Reset the seed used to generate stateful random numbers and perform
  stochastic rounding.

  Args:
    seed: The new random number generator seed.
    device: The device to which the seed will be applied.
    cpu_device: The CPU device which is on the same hardware to the IPU device.
    experimental_identical_replicas: Whether to seed all the local replicas
      identically. Note that to generate identical sequences of random numbers
      on all replicas, the Poplar engine option `"target.deterministicWorkers"`
      must also be set to `"portable"`. Also note that for multi-replica
      distribution with multiple processes, the same seed must be passed to
      each process to ensure that all the replicas globally get the same seed.
      WARNING: This flag is experimental and subject to change.

  Returns:
    None
  """
  g = ops.Graph()
  with g.as_default():
    with ops.device(cpu_device):
      cfg_op = gen_ipu_ops.ipu_reset_seed(
          device, seed, identical_replicas=experimental_identical_replicas)

  with session_lib.Session(graph=g) as sess:
    sess.run(cfg_op)


def extract_all_strings_from_event_trace(events):
  """Extract a concatenation of all data strings from an IPU event trace.

  Args:
    events: An array of IPU events as returned from the ``ipu_compile_summary``
      operation.

  Returns:
    A string containing the concatenation of all of the data fields of the
    events.

  """
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)

    result = result + ("-" * 70) + "\n=> @ " + \
             time.strftime('%F %T %z', time.localtime(evt.timestamp)) + ": "

    if evt.type == IpuTraceEvent.COMPILE_BEGIN:
      evt_str = "Compile begin: " + \
                evt.compile_begin.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.COMPILE_END:
      evt_str = "Compile end: " + \
                evt.compile_end.module_name.decode('utf-8') + "\n" + \
                "Duration: " + str(evt.compile_end.duration)
    elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
      evt_str = "Host->Device\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
      evt_str = "Device->Host\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.LOAD_ENGINE:
      evt_str = "Load engine: " + \
                evt.load_engine.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.EXECUTE:
      evt_str = "Execute: " + \
                evt.execute.module_name.decode('utf-8')
    else:
      evt_str = "Unknown event"

    result = result + evt_str + '\n'

  return result


def extract_all_types_from_event_trace(events):
  """Return a list of the types of each event in an event trace tensor

  Args:
    events: A tensor containing a list of IPU events as protobuf strings

  Returns:
    A list containing the type of each event
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt.type]
  return result


def extract_all_events(events):
  """Extract a list containing each event as an event object

  Args:
    events: A tensor containing a list of IPU events as protobuf strings

  Returns:
    A list containing IpuTraceEvent objects
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt]
  return result


@deprecation.deprecated(None, "Poplar reports are no longer included in "
                        "trace events. You can use the PopVision Graph "
                        "Analyser for manual inspection of reports.")
def extract_compile_reports(events):
  """Get a list of all compiler reports in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  return []


@deprecation.deprecated(None, "Poplar reports are no longer included in "
                        "trace events. You can use the PopVision Graph "
                        "Analyser for manual inspection of reports.")
def extract_poplar_serialized_graphs(events):
  """Get a list of all Poplar serialized graphs in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  return []


@deprecation.deprecated(None, "Poplar reports are no longer included in "
                        "trace events. You can use the PopVision Graph "
                        "Analyser for manual inspection of reports.")
def extract_execute_reports(events):
  """Get a list of all compiler reports in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  return []


def move_variable_initialization_to_cpu(graph=None):
  """For all variables in the VARIABLES collection, move any initialization
  ops onto the CPU.

  Args:
    graph: Operations are moved around on this graph.  The default graph will be
           used if not specified.

  Returns:
    None
  """
  if not graph:
    graph = ops.get_default_graph()

  with ops.device("/device:CPU:0"):
    control_flow_ops.no_op(name="cpu")
  variables = []
  for v in graph.get_collection('variables'):
    # We assume a distribution strategy knows better how to
    # initialize its own variables, so skip those.
    if not isinstance(v, values.DistributedVariable):
      variables.append(v)

  def _uses_resource(op):
    """ Helper to determine if an op uses a resource """
    return any(input_tensor.dtype == 'resource' for input_tensor in op.inputs)

  init_ops = []
  dep_ops = [v.initializer.inputs[1].op for v in variables]
  visited = set()

  # Depth-first search up the graph starting from all variables in VARIABLES
  # Place all touched ops on the CPU, but do not touch or search ops that use
  # resource tensors, otherwise device colocation could be violated.
  while dep_ops:
    op = dep_ops.pop()
    if op not in visited and not _uses_resource(op):
      visited.add(op)
      init_ops += [op]
      dep_ops += [x.op for x in op.inputs]

  # pylint: disable=protected-access
  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr(
        '_class',
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
            s=[b'loc:@cpu'])))
    op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
    op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
  # pylint: enable=protected-access

  return


def export_dataset_to_file(dataset_or_infeed,
                           output_filename,
                           num_elements,
                           feed_name="",
                           apply_options=True):
  """Export as binary `num_elements` from the given `infeed` to the specified
  `output_filename`.

  If the infeed elements are tuples then one file per tuple element will be
  created.
  For example, if `dataset` looks like

  .. code-block:: python

    [{ "a": A_0, "b": B_0}, { "a": A_1, "b": B_1}, ...]

  then `export_dataset_to_file(dataset, "my_dataset.bin", 100)` will generate:

  .. code-block:: python

    my_dataset.0.bin   # Contains tensors [ A_0, A_1, ..., A_99]
    my_dataset.1.bin   # Contains tensors [ B_0, B_1, ..., B_99]

  Args:
    dataset_or_infeed: An unary dataset with the same input and output
      structure or an `IPUInfeedQueue`.
    output_filename: Where to export the tensors to.
    num_elements: Number of elements to export from the dataset.
    feed_name: Specify the feed name.
    apply_options: Whether to apply optimization options which can improve the
      dataset performance.
  """
  assert isinstance(dataset_or_infeed,
                    (dataset_ops.Dataset, ipu_infeed_queue.IPUInfeedQueue))
  if isinstance(dataset_or_infeed, ipu_infeed_queue.IPUInfeedQueue):
    dataset = dataset_or_infeed._dataset  # pylint: disable=protected-access
    feed_name = feed_name or dataset_or_infeed._id  # pylint: disable=protected-access
  else:
    dataset = dataset_or_infeed
  if apply_options:
    dataset = dataset._apply_options()  # pylint: disable=protected-access

  extractor = dataset_extractor.dataset_extractor(dataset, num_elements,
                                                  output_filename, feed_name)
  with ops.device("cpu"), session_lib.Session() as sess:
    sess.run(extractor)


def export_inputs_to_file(inputs, output_filename, feed_dict):
  """Export as binary the list of `inputs` provided to the specified
  `output_filename`.

  Args:
    inputs: List of graph inputs to export.
    output_filename: Where to export the tensors to.
    feed_dict: Feed dictionary containing the inputs' values.
  """

  with ops.device("cpu"), session_lib.Session() as sess:
    sess.run(dataset_extractor.export_variables(inputs, output_filename),
             feed_dict)


def use_synthetic_data_for(synthetic_data_category):
  """Get whether synthetic data is being used for the given category.

  Args:
    synthetic_data_category: A SyntheticDataCategory enum value.

  Returns:
    A bool indicating the result.
  """

  op = gen_ipu_ops.ipu_use_synthetic_data_for(
      synthetic_data_category=synthetic_data_category)

  if executing_eagerly():
    return op.numpy()[0]
  return session_lib.Session().run(op)[0]

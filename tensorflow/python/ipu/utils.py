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
General utility functions
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import collections
from enum import Enum
import os
import time
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
# pylint: disable=unused-import
from tensorflow.compiler.plugin.poplar.tools.tensorflow_weights_extractor import export_variables_from_live_session, export_variables_from_live_model
# pylint: enable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import values
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import dataset_extractor


class SelectionOrder(Enum):
  """Depending on the communication pattern of the model, the order in
  which the IPUs are selected and mapped to shards can impact the performance.

  For example, given a model which executes on multiple IPUs:

  .. code-block:: python

    def sharded_graph(pa, pb, pc, pd):
      with ipu.scopes.ipu_shard(0):
        o1 = pa + pb
      with ipu.scopes.ipu_shard(1):
        o2 = o1 + pc
      with ipu.scopes.ipu_shard(2):
        o3 = o2 + pd
        return o3

  and a typical machine with 8 Graphcore C2 cards:

  .. code-block:: none

     _______               _______
    |       |             |       |
    |  14   |=============|  15   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |  12   |=============|  13   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |  10   |=============|  11   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   8   |=============|   9   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   6   |=============|   7   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   4   |=============|   5   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   2   |=============|   3   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   0   |=============|   1   |
    |_______|             |_______|

  (where each numbered square represents an IPU with the given device ID and the
  == and || connections represent IPUs being directly connected via IPU-Links)

  we can see that the `ipu_shard(0)` directly communicates with `ipu_shard(1)`
  and that `ipu_shard(1)` directly communicates with `ipu_shard(2)`.
  If the shards 0, 1, 2 were mapped to IPUs 0, 1, 2 in that order, then the
  communication between shards 1 and 2 would not have a direct connection via an
  IPU-Link and would have to perform a "hop" via an IPU.
  If the shards 0, 1, 2 were mapped to IPUs 0, 1, 3 in that order, then the
  communication between shards 1 and 2 would have a direct connection via an
  IPU-Link which will reduce the communication cost.

  This Enum class is used to control the order in which the IPUs are selected.
  Currently, the following IPU selection orderings are supported:
  * `AUTO`: automatically try and select the best selection given the network.
  * `ZIGZAG`: follow the natural ordering of IPUs. In the above example, the
    IPUs would be selected in the following order:
    `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`.
  * `SNAKE`: select IPUs such that each consecutive shard is directly
    connected via IPU-Links to the shard before and after. In the above example,
    the IPUs would be selected in the following order:
    `0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14`.
  * `HOOF`: select IPUs such that each consecutive shard is directly
    connected via IPU-Links to the shard before and after and the last and first
    shard are on the same C2 cards. In the above example, the IPUs would be
    selected in the following order:
    `0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3, 1`.

  The `SNAKE` and `HOOF` IPU selection orders are particularly beneficial for
  pipelined models.
  """
  AUTO = config_pb2.IpuSelectionOrder.Value("AUTO")
  ZIGZAG = config_pb2.IpuSelectionOrder.Value("ZIGZAG")
  SNAKE = config_pb2.IpuSelectionOrder.Value("SNAKE")
  HOOF = config_pb2.IpuSelectionOrder.Value("HOOF")


class ExecutionProfileType(Enum):
  """The execution profile type indicates the desired information in the
  execution profile.

  * `NO_PROFILE` indicates that there should be no execution profiling.
  * `DEVICE_PROFILE` indicates that the execution profile should contain only
    device wide events.
  * `IPU_PROFILE` indicates that the profile should contain IPU level
    execution events.
  * `TILE_PROFILE` indicates that the profile should contain Tile level
    execution events.
  """
  NO_PROFILE = config_pb2.IpuExecutionProfileType.Value("NO_PROFILE")
  DEVICE_PROFILE = config_pb2.IpuExecutionProfileType.Value("DEVICE_PROFILE")
  IPU_PROFILE = config_pb2.IpuExecutionProfileType.Value("IPU_PROFILE")
  TILE_PROFILE = config_pb2.IpuExecutionProfileType.Value("TILE_PROFILE")


class DeviceConnectionType(Enum):
  """Enumeration to describe the mechanism used to attach to the Poplar
  device.

  * `ALWAYS` indicates that the system will attach when configuring the
    device.
  * `ON_DEMAND` will defer connection to when the IPU is needed.
  * `NEVER` will never try to attach to a device. Used when compiling offline.
  """
  ALWAYS = config_pb2.IpuDeviceConnectionType.Value("ALWAYS")
  ON_DEMAND = config_pb2.IpuDeviceConnectionType.Value("ON_DEMAND")
  NEVER = config_pb2.IpuDeviceConnectionType.Value("NEVER")


def configure_ipu_system(config, device="cpu"):
  """Configure an IPU system.  Passing an IpuOptions protobuf created by the
  ``create_ipu_config`` function.

  Args:
    config: An IpuOptions configuration protobuf
    device: The CPU device which is local to the IPU hardware

  Returns:
    None
  """
  if not isinstance(config, config_pb2.IpuOptions):
    raise Exception("`config` must be an IpuOptions instance")

  g = ops.Graph()
  with g.as_default():
    with ops.device(device):
      cfg_op = gen_ipu_ops.ipu_configure_hardware(config.SerializeToString())

  with session_lib.Session(graph=g) as sess:
    sess.run(cfg_op)


def running_on_ipu_model():
  """ Check if XLA is configured to run on the ipu model.

  Returns:
    True if XLA is configured to run on the ipu model.
    False if XLA is configured to run on real hardware.
  """
  return "--use_ipu_model" in os.environ.get("TF_POPLAR_FLAGS", "")


@deprecation.deprecated_args(None, "Use set_optimization_options() instead.",
                             "max_cross_replica_sum_buffer_size",
                             "max_inter_ipu_copies_buffer_size")
def create_ipu_config(profiling=False,
                      enable_ipu_events=False,
                      use_poplar_text_report=False,
                      use_poplar_cbor_report=False,
                      profile_execution=None,
                      enable_poplar_serialized_graph=False,
                      report_every_nth_execution=0,
                      max_report_size=0x10000000,
                      report_directory="",
                      scheduler_selection="",
                      always_rearrange_copies_on_the_host=False,
                      merge_infeed_io_copies=False,
                      disable_graph_convolution_caching=False,
                      disable_graph_outlining=False,
                      retain_control_dependencies=False,
                      max_cross_replica_sum_buffer_size=0,
                      max_inter_ipu_copies_buffer_size=0,
                      max_scheduler_lookahead_depth=5,
                      max_scheduler_search_space_size=64,
                      prefetch_data_streams=True,
                      selection_order=None):
  """Create an empty IPU session configuration structure.

  Args:
    profiling: Enable compilation reports, and IPU trace events.
    enable_ipu_events: Enable IPU trace events without poplar reports.
    use_poplar_text_report: Enable the poplar textual report summary
    use_poplar_cbor_report: Enable the poplar CBOR reports
    profile_execution: Include Poplar execution profiles in the execution
      events. Can only be enabled if `profling` is also enabled. If set, can be
      `True`, 'False`, or a member of the `ExecutionProfileType` enumeration.
      A `True` value indicates `ExecutionProfileType.DEVICE_PROFILE`.
    include_poplar_serialized_graph: Create the Poplar serialized graph and
      include in the IPU compilation trace events.
    report_every_nth_execution: Only produce an execution report on every Nth
      execution.  0 = One report only.
    max_report_size: The maximum size of Poplar profiles to include in the
      profile events.
    report_directory: When set, reports will be written to files in this
      directory, instead of being written into the events.  The events will
      contain the full paths of the report files.
    scheduler_selection: When set, this forces the compiler to use a specific
      scheduler when ordering the instructions.  See the documentation for a
      list of valid schedulers.
    always_rearrange_copies_on_the_host: *** Experimental Flag ***
      The data which is streamed to/from the device might be stored in different
      layouts on the device and on the host. If that is the case the
      rearrangment is performed on the device by default. By enabling this
      option the rearrangment will be performed on the host at the expense of
      latency.
    merge_infeed_io_copies: When true, this flag will merge the streamed
      host->device input copies into one larger copy.  This may reduce the time
      to copy data from the host, at the expense of increasing the live tensor
      memory on the device.
    disable_graph_convolution_caching: By default, the convolution operation
      searches for an equivalent cached operation, and uses this  instead of
      creating a new convolution. Setting this flag forces the creation of a
      new convolution. This can improve runtime at the expense of graph size.
    disable_graph_outlining: By default, some operations, such as matrix
      multiplications, which occur in the graph multiple times but with
      different input tensors might be optimised to reduce the total code size
      of the graph at the expense of the execution time. Setting this flag will
      disable these optimisations. This option is not valid for the convolution
      operation (also see disable_graph_convolution_caching)
    retain_control_dependencies: Deprecated.
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
      IPU devices when using a multi-IPU devices (see `SelectionOrder`). When
      not specified, then automatic selection order is used, otherwise an
      instance of `SelectionOrder`.

  Returns:
    An IpuOptions configuration protobuf, suitable for passing to
    ``configure_ipu_system``
  """
  if profiling and enable_ipu_events:
    raise Exception(
        "`profiling` and `enable_ipu_events` are mutually exclusive")

  if retain_control_dependencies:
    raise Exception("`retain_control_dependencies` is deprecated")

  selection_order = selection_order if selection_order else SelectionOrder.AUTO
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
  opts.enable_multi_slice_combiner = False
  opts.enable_matmul_combiner = False
  opts.enable_gather_simplifier = False
  opts.device_connection_type = DeviceConnectionType.ALWAYS.value
  opts.speed_size_config.allow_recompute = False
  opts.speed_size_config.allow_stateful_recompute = False

  # Configure IpuOptions according to the passed arguments.
  opts.profiling.enable_ipu_trace_events = profiling or enable_ipu_events
  opts.profiling.enable_compilation_trace = profiling
  opts.profiling.enable_io_trace = profiling
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
  opts.speed_size_config.disable_graph_convolution_caching = \
      disable_graph_convolution_caching
  opts.speed_size_config.disable_graph_outlining = \
      disable_graph_outlining
  opts.speed_size_config.scheduler_selection = scheduler_selection

  opts.max_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size
  opts.max_inter_ipu_copies_buffer_size = max_inter_ipu_copies_buffer_size

  opts.max_scheduler_lookahead_depth = max_scheduler_lookahead_depth
  opts.max_scheduler_search_space_size = max_scheduler_search_space_size

  opts.prefetch_data_streams = prefetch_data_streams
  opts.selection_order = selection_order.value

  opts.verified_transfers.enabled = False
  opts = set_verification_options(opts, VerificationOptions())

  return opts


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


def set_optimization_options(opts,
                             combine_embedding_lookups=False,
                             combine_matmuls=False,
                             max_cross_replica_sum_buffer_size=0,
                             max_reduce_scatter_buffer_size=0,
                             max_inter_ipu_copies_buffer_size=0,
                             max_send_recv_cluster_size=0,
                             gather_simplifier=False):
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
    gather_simplifier: Will enable more aggressive optimisation
      for embedding lookups.

  Returns:
    The IpuOptions configuration protobuf.
  """
  # Internally embedding lookups are implemented using multiSlice operations.
  opts.enable_multi_slice_combiner = combine_embedding_lookups
  opts.enable_matmul_combiner = combine_matmuls
  opts.max_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size
  opts.max_reduce_scatter_buffer_size = max_reduce_scatter_buffer_size
  opts.max_inter_ipu_copies_buffer_size = max_inter_ipu_copies_buffer_size
  opts.max_send_recv_cluster_size = max_send_recv_cluster_size
  opts.enable_gather_simplifier = gather_simplifier

  return opts


def set_norm_options(opts, use_stable_statistics=False):
  """Set the IPU options related to norms.

  Args:
    use_stable_statistics: If True, computes the mean first and subtracts
      the activations by it before computing the variance. The
      implementation with this flag set to True is slower than when set
      to False.

  Returns:
    The IpuOptions configuration protobuf.
  """
  opts.use_stable_norm_statistics = use_stable_statistics

  return opts


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


class KeyId:
  def __init__(self, key=0, start_id=-1):
    self.key = key
    self.start_id = start_id


class VerificationOptions:
  """Store pairs of key / id to use for each type of data used in the graph.
  Does nothing unless verified transfers have been enabled by calling
  `set_transfer_options(opts, use_verified_transfers=True).`
  And an instance of this class has been set by calling
  `set_verification_options`:

  .. code-block:: python
    o = VerificationOptions()
    o.inputs.key = 1
    o.infeeds["infeed"].key = 3
    set_verification_options(opts, o)
  """
  def __init__(self):
    self.inputs = KeyId()
    self.input_parameters = KeyId()
    self.outputs = KeyId()
    self.output_parameters = KeyId()
    self.infeeds = collections.defaultdict(KeyId)
    self.outfeeds = collections.defaultdict(KeyId)
    self.checkpoint_in = KeyId(0, 0)
    self.checkpoint_out = KeyId(0, 0)


def set_verification_options(opts, verification_options):
  """Set the pairs or key / id to use for each type of data used in the graph
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
    compilation_options: A dictionary of poplar compilation option flags to be
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
    convolution_options: A dictionary of poplar option flags for
      convolutions. The "availableMemoryProportion" flag indicates the
      proportion of tile memory to be made available asß
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
    matmul_options: A dictionary containing the poplar option flag
      "availableMemoryProportion" for the matrix multiplication operations.
      It indicates the proportion of tile memory to be made available as
      temporary memory for the matrix multiplications (float between 0 and 1.0).
      Less temporary memory will generally result in a multiplication that
      takes more cycles to complete. However, because always live memory
      (like code and vertex state) is not tracked when planning it,
      a multiplication using less temporary memory may use more memory overall,
      due to an increase of always live memory.
    clear_pass_type: When set to True, the Pass type will not
      be set in the options passed to the poplar operation.

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
    pooling_options: A dictionary of poplar option flags for the pooling
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


@deprecation.deprecated_args(
    None, "report_options is deprecated, use graph_options and"
    " execution_options instead", "report_options")
def set_report_options(opts,
                       report_options=None,
                       graph_options=None,
                       execution_options=None):
  """Set the options used to influence Poplar graph and execution reports
     generation.


  .. code-block:: python

      opts = create_ipu_config()
      opts = set_report_options(opts,
          report_options={"reportOption1": "false"},
          graph_options={"graphOptions": "false"},
          execution_options={"executionOptions": "false"})
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  Args:
    opts: An IpuOptions session control protobuf.
    report_options: (Deprecated) A dictionary of poplar option flags for
      the report generation.
    graph_options: A dictionary of poplar option flags for the graph report
      generation.
    execution_options: A dictionary of poplar option flags for the execution
      report generation.

  Returns:
    The IpuOptions configuration protobuf, with convolution options set.
  """
  def use_report_options():
    if report_options:
      if not isinstance(report_options, dict):
        raise Exception("`report_options` must be a dictionary")
    return report_options

  if not graph_options:
    graph_options = use_report_options()

  if graph_options:
    if not isinstance(graph_options, dict):
      raise Exception("`graph_options` must be a dictionary")

    for (option_name, value) in graph_options.items():
      opt = opts.profiling.graph_options.add()
      opt.option = option_name
      opt.value = value

  if not execution_options:
    execution_options = use_report_options()

  if execution_options:
    if not isinstance(execution_options, dict):
      raise Exception("`execution_options` must be a dictionary")

    for (option_name, value) in execution_options.items():
      opt = opts.profiling.execution_options.add()
      opt.option = option_name
      opt.value = value

  return opts


def set_ipu_model_options(opts, compile_ipu_code=True):
  """Set the IPU Model options.

  Args:
    compile_ipu_code: Whether or not to actually compile real IPU code for
      modelling.

  Returns:
    The IpuOptions configuration protobuf, with IPU model options set.
  """
  opts.ipu_model_config.compile_ipu_code = compile_ipu_code

  return opts


def set_recomputation_options(opts,
                              allow_recompute=True,
                              allow_stateful_recompute=True):
  """Set re-computation options.

  Args:
    allow_recompute: Whether or not to re-compute instructions during training.
      If this is enabled then we will attempt to pattern match
      instructions/pipeline stages in the forward pass and recompute them in the
      backward pass to avoid having to preserve activations which increase the
      maximum memory liveness. Enabling this option can reduce memory usage at
      the expense of extra computation.
    allow_stateful_recompute: Whether or not to extend the re-compute of
      pipeline stages to stages containing stateful operations (Has no effect
      if allow_recompute is False).

  Returns:
    The IpuOptions configuration protobuf.
  """

  opts.speed_size_config.allow_recompute = allow_recompute
  opts.speed_size_config.allow_stateful_recompute = allow_stateful_recompute

  return opts


def set_floating_point_behaviour_options(opts,
                                         inv=True,
                                         div0=True,
                                         oflo=True,
                                         esr=True,
                                         nanoo=True):
  """Set the IPU floating point control behaviour bits

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
  opts.floating_point_behaviour.flags_set = True
  opts.floating_point_behaviour.inv = inv
  opts.floating_point_behaviour.div0 = div0
  opts.floating_point_behaviour.oflo = oflo
  opts.floating_point_behaviour.esr = esr
  opts.floating_point_behaviour.nanoo = nanoo

  return opts


def auto_select_ipus(opts, num_ipus):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple Tensorflow
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
    num_ipus: List of IPUs per Tensorflow device

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


def select_ipus(opts, indices):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple Tensorflow
  devices, each with control of one of more IPUs. The Tensorflow devices will be
  labeled ``/device:IPU:0``, ``/device:IPU:1`` and so on.

  Each Tensorflow device uses a specific configuration consisting of one or more
  IPUs from the list of devices.  These can be found by running the Graphcore
  utility ``gc-info -l``.  For instance, the following listing shows the device
  configurations available on a system with 16 IPUs.

  .. code-block:: shell

      user@host:~$ gc-info -l
      Graphcore device listing:

      -+- Id:  [0], type:      [PCIe], PCI Domain: [0000:1a:00.0]
      -+- Id:  [1], type:      [PCIe], PCI Domain: [0000:1b:00.0]
      -+- Id:  [2], type:      [PCIe], PCI Domain: [0000:1c:00.0]
      -+- Id:  [3], type:      [PCIe], PCI Domain: [0000:1d:00.0]
      -+- Id:  [4], type:      [PCIe], PCI Domain: [0000:60:00.0]
      -+- Id:  [5], type:      [PCIe], PCI Domain: [0000:61:00.0]
      -+- Id:  [6], type:      [PCIe], PCI Domain: [0000:62:00.0]
      -+- Id:  [7], type:      [PCIe], PCI Domain: [0000:63:00.0]
      -+- Id:  [8], type:      [PCIe], PCI Domain: [0000:b1:00.0]
      -+- Id:  [9], type:      [PCIe], PCI Domain: [0000:b2:00.0]
      -+- Id: [10], type:      [PCIe], PCI Domain: [0000:b3:00.0]
      -+- Id: [11], type:      [PCIe], PCI Domain: [0000:b4:00.0]
      -+- Id: [12], type:      [PCIe], PCI Domain: [0000:da:00.0]
      -+- Id: [13], type:      [PCIe], PCI Domain: [0000:db:00.0]
      -+- Id: [14], type:      [PCIe], PCI Domain: [0000:dc:00.0]
      -+- Id: [15], type:      [PCIe], PCI Domain: [0000:dd:00.0]
      -+- Id: [32], type: [Multi IPU]
       |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
       |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
       |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
       |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
       |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:1d:00.0]
       |--- PCIe Id:  [2], DNC Id: [5], PCI Domain: [0000:1c:00.0]
       |--- PCIe Id:  [1], DNC Id: [6], PCI Domain: [0000:1b:00.0]
       |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
       |--- PCIe Id: [11], DNC Id: [8], PCI Domain: [0000:b4:00.0]
       |--- PCIe Id: [10], DNC Id: [9], PCI Domain: [0000:b3:00.0]
       |--- PCIe Id:  [9], DNC Id: [10], PCI Domain: [0000:b2:00.0]
       |--- PCIe Id:  [8], DNC Id: [11], PCI Domain: [0000:b1:00.0]
       |--- PCIe Id: [15], DNC Id: [12], PCI Domain: [0000:dd:00.0]
       |--- PCIe Id: [14], DNC Id: [13], PCI Domain: [0000:dc:00.0]
       |--- PCIe Id: [13], DNC Id: [14], PCI Domain: [0000:db:00.0]
       |--- PCIe Id: [12], DNC Id: [15], PCI Domain: [0000:da:00.0]
      -+- Id: [33], type: [Multi IPU]
       |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
       |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
       |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
       |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
       |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:1d:00.0]
       |--- PCIe Id:  [2], DNC Id: [5], PCI Domain: [0000:1c:00.0]
       |--- PCIe Id:  [1], DNC Id: [6], PCI Domain: [0000:1b:00.0]
       |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
      -+- Id: [34], type: [Multi IPU]
       |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
       |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
       |--- PCIe Id:  [9], DNC Id: [2], PCI Domain: [0000:b2:00.0]
       |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:b1:00.0]
       |--- PCIe Id: [15], DNC Id: [4], PCI Domain: [0000:dd:00.0]
       |--- PCIe Id: [14], DNC Id: [5], PCI Domain: [0000:dc:00.0]
       |--- PCIe Id: [13], DNC Id: [6], PCI Domain: [0000:db:00.0]
       |--- PCIe Id: [12], DNC Id: [7], PCI Domain: [0000:da:00.0]
      -+- Id: [35], type: [Multi IPU]
       |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
       |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
       |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
       |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
      -+- Id: [36], type: [Multi IPU]
       |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:1d:00.0]
       |--- PCIe Id:  [2], DNC Id: [1], PCI Domain: [0000:1c:00.0]
       |--- PCIe Id:  [1], DNC Id: [2], PCI Domain: [0000:1b:00.0]
       |--- PCIe Id:  [0], DNC Id: [3], PCI Domain: [0000:1a:00.0]
      -+- Id: [37], type: [Multi IPU]
       |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
       |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
       |--- PCIe Id:  [9], DNC Id: [2], PCI Domain: [0000:b2:00.0]
       |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:b1:00.0]
      -+- Id: [38], type: [Multi IPU]
       |--- PCIe Id: [15], DNC Id: [0], PCI Domain: [0000:dd:00.0]
       |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:dc:00.0]
       |--- PCIe Id: [13], DNC Id: [2], PCI Domain: [0000:db:00.0]
       |--- PCIe Id: [12], DNC Id: [3], PCI Domain: [0000:da:00.0]
      -+- Id: [39], type: [Multi IPU]
       |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
       |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
      -+- Id: [40], type: [Multi IPU]
       |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:61:00.0]
       |--- PCIe Id:  [4], DNC Id: [1], PCI Domain: [0000:60:00.0]
      -+- Id: [41], type: [Multi IPU]
       |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:1d:00.0]
       |--- PCIe Id:  [2], DNC Id: [1], PCI Domain: [0000:1c:00.0]
      -+- Id: [42], type: [Multi IPU]
       |--- PCIe Id:  [1], DNC Id: [0], PCI Domain: [0000:1b:00.0]
       |--- PCIe Id:  [0], DNC Id: [1], PCI Domain: [0000:1a:00.0]
      -+- Id: [43], type: [Multi IPU]
       |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
       |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
      -+- Id: [44], type: [Multi IPU]
       |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:b2:00.0]
       |--- PCIe Id:  [8], DNC Id: [1], PCI Domain: [0000:b1:00.0]
      -+- Id: [45], type: [Multi IPU]
       |--- PCIe Id: [15], DNC Id: [0], PCI Domain: [0000:dd:00.0]
       |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:dc:00.0]
      -+- Id: [46], type: [Multi IPU]
       |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:db:00.0]
       |--- PCIe Id: [12], DNC Id: [1], PCI Domain: [0000:da:00.0]

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

      # Create a single device with 1 IPU at PCI address 0000:b1:00.0 by using
      # IPU configuration index 8
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[8])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create two Tensorflow devices, with one IPU each, being devices at
      # indices 0 and 1
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[0, 1])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create two Tensorflow devices, with four IPUs each. The device
      # configurations at indices 37 (0000:b4:00.0, 0000:b3:00.0, 0000:b2:00.0,
      # 000:b1:00.0) and 38 (0000:dd:00.0, 0000:dc:00.0, 0000:db:00.0,
      # 00:da:00.0)
      opts = create_ipu_config()
      opts = select_ipus(opts, indices=[37, 38])
      ipu.utils.configure_ipu_system(opts)
      with tf.Session() as s:
        ...

  .. code-block:: python

      # Create four Tensorflow devices each with one IPU, at addresses
      # 0000:1a:00.0, 0000:1b:00.0, 0000:1c:00.0, 0000:1d:00.0.
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
    raise Exception("All device indeicies in `indices` must be unique.")

  for i in indices:
    dev = opts.device_config.add()
    dev.cfg_index = i

  return opts


def set_ipu_connection_type(opts, connection_type=None, ipu_version=None):
  """ Configure when to attach to the device.

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
                     Defaults to `DeviceConnectionType.ALWAYS` if None.

    ipu_version: Version of the IPU hardware used. Required if the
                 `connection_type` provided is `DeviceConnectionType.NEVER`.
  Returns:
    The IpuOptions configuration protobuf.
  """
  connection_type = connection_type if connection_type \
                                    else DeviceConnectionType.ALWAYS

  if connection_type == DeviceConnectionType.NEVER and ipu_version is None:
    raise Exception("`ipu_version` must be set when `connection_type` is set "
                    "to `DeviceConnectionType.NEVER`")
  opts.device_connection_type = connection_type.value

  if ipu_version is not None:
    opts.ipu_version = ipu_version
    opts.has_ipu_version = True

  return opts


def reset_ipu_seed(seed, device="/device:IPU:0", cpu_device="cpu"):
  """Reset the seed used to generate stateful random numbers and perform
  stochastic rounding.

  Args:
    seed: The new random number generator seed.
    device: The device to which the seed will be applied.
    cpu_device: The CPU device which is on the same hardware to the IPU device.

  Returns:
    None
  """
  g = ops.Graph()
  with g.as_default():
    with ops.device(cpu_device):
      cfg_op = gen_ipu_ops.ipu_reset_seed(device, seed)

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
                "Duration: " + str(evt.compile_end.duration) + " us\n" + \
                evt.compile_end.compilation_report.decode('utf-8')
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
                evt.execute.module_name.decode('utf-8') + "\n" + \
                evt.execute.execution_report.decode('utf-8')
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


def extract_compile_reports(events):
  """Get a list of all compiler reports in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.COMPILE_END:
      try:
        module = evt.compile_end.module_name.decode('utf-8')
        rep = evt.compile_end.compilation_report.decode('utf-8')
        if rep:
          result += [(module, rep)]
      except UnicodeDecodeError:
        pass
  return result


def extract_poplar_serialized_graphs(events):
  """Get a list of all poplar serialized graphs in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.COMPILE_END:
      try:
        rep = evt.compile_end.poplar_graph.decode('utf-8')
      except UnicodeDecodeError:
        rep = evt.compile_end.poplar_graph

      module = evt.compile_end.module_name.decode('utf-8')
      if rep:
        result += [(module, rep)]
  return result


def extract_execute_reports(events):
  """Get a list of all compiler reports in the event list.

  Args:
    events: A list of trace event serialized protobufs.

  Returns:
    A list of tuples containing the module name and report."""
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.EXECUTE:
      try:
        module = evt.execute.module_name.decode('utf-8')
        rep = evt.execute.execution_report.decode('utf-8')
        if rep:
          result += [(module, rep)]
      except UnicodeDecodeError:
        pass
  return result


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

  init_ops = []
  dep_ops = [v.initializer.inputs[1].op for v in variables]
  visited = set()

  while dep_ops:
    op = dep_ops.pop()
    if not op in visited:
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
  For example if `dataset` looks like
  [{ "a": A_0, "b": B_0}, { "a": A_1, "b": B_1}, ...]
  Then `export_dataset_to_file(dataset, "my_dataset.bin", 100)` will generate:
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

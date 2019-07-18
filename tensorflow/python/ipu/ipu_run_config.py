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
"""A RunConfig subclass with IPU support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.estimator import run_config as run_config_lib


class IPURunConfig(
    collections.namedtuple('IPURunConfig', [
        'iterations_per_loop', 'ipu_options', 'compile_summary',
        'num_replicas', 'num_shards', 'autosharding'
    ])):
  r"""IPU related configuration required by `IPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps running in IPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
      Note that in evaluation don't use this value, instead we run total eval
      `steps` on IPU for a single `Session.run`.
    ipu_options: An IpuOptions configuration protobuf which is populated prior
      to being passed into IPURunConfig. Note that if more than one device is
      being used then `ipu_options` needs to be populated with a `device_config`.
    compile_summary: Generate compilation summary
    num_replicas: Number of replicated graphs(data parallel)
    num_shards: Number of IPU devices the on which the graph is sharded (model parallel)
    autosharding: Use the IPU `automatic_sharding` to automatically shard the graph
      across `num_shards` devices
  """

  def __new__(cls,
              iterations_per_loop=1,
              ipu_options=None,
              compile_summary=False,
              num_replicas=1,
              num_shards=1,
              autosharding=False):

    num_devices = num_replicas * num_shards
    if num_devices > 1 and ipu_options is None:
      raise ValueError(
          'IPU configuration requires more than one device, but `ipu_options` is None'
      )

    num_configured_devices = 1
    if ipu_options is not None:
      if len(ipu_options.device_config
             ) == 1 and ipu_options.device_config[0].auto_count > 0:
        num_configured_devices = ipu_options.device_config[0].auto_count
      elif len(ipu_options.device_config) > 1:
        num_configured_devices = len(ipu_options.device_config)

    if num_devices != num_configured_devices:
      raise ValueError(
          '`IpuOptions` configured with {} devices, but `IPURunConfig` configured with {} devices'
          .format(num_configured_devices, num_devices))

    return super(IPURunConfig,
                 cls).__new__(cls,
                              iterations_per_loop=iterations_per_loop,
                              ipu_options=ipu_options,
                              compile_summary=compile_summary,
                              num_replicas=num_replicas,
                              num_shards=num_shards,
                              autosharding=autosharding)


class RunConfig(run_config_lib.RunConfig):
  """RunConfig with IPU support."""

  def __init__(self, ipu_run_config=None, **kwargs):
    """Constructs a RunConfig.

    Args:
      ipu_run_config: the IPURunConfig that specifies IPU-specific configuration.
      **kwargs: keyword config parameters.
    """
    super(RunConfig, self).__init__(**kwargs)
    self._ipu_run_config = ipu_run_config or IPURunConfig()

  @property
  def ipu_run_config(self):
    return self._ipu_run_config

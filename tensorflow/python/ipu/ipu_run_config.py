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
Run configs
~~~~~~~~~~~
"""
import collections

from tensorflow.python.estimator import run_config as run_config_lib


class IPURunConfig(
    collections.namedtuple('IPURunConfig', [
        'iterations_per_loop', 'ipu_options', 'compile_summary',
        'num_replicas', 'num_shards', 'autosharding'
    ])):
  """IPU related configuration required by `IPUEstimator`.

  Args:
    iterations_per_loop: This is the number of iterations running on the IPU device
      before returning to the CPU host for each `Session.run`. This means that the
      global step is increased `iterations_per_loop` times in one `Session.run`.
    ipu_options: An IpuOptions configuration protobuf which is populated prior
      to being passed into IPURunConfig. Note that if more than one device is
      being used then `ipu_options` needs to be populated with a `device_config`.
    compile_summary: Generate compilation summary
    num_replicas: Number of replicated graphs (data parallelism)
    num_shards: Number of IPU devices on which the graph is sharded (model parallelism)
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
          '`IPURunConfig` configured with {} devices ({} num_replicas times {} num_shards),'
          ' but `IpuOptions` configured with {} devices'.format(
              num_devices, num_replicas, num_shards, num_configured_devices))

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
  def __init__(self, ipu_run_config=None, master=None, **kwargs):
    """Constructs a RunConfig with IPU support.

    These are the arguments specific to the RunConfig for IPUs. All remaining
    keyword arguments are passed to the base class, which is documented below.

    Args:
      ipu_run_config: :class:`.IPURunConfig` object for IPU-specific
        configuration.
      master: a string. The address of the distributed master to use for
        training.
    """
    super(RunConfig, self).__init__(**kwargs)
    self._ipu_run_config = ipu_run_config or IPURunConfig()

    # master is set by the parent class based on TF_CONFIG,
    # only override here if the user gave it explicitly.
    if master is not None:
      self._master = master

  __init__.__doc__ += run_config_lib.RunConfig.__init__.__doc__

  @property
  def ipu_run_config(self):
    return self._ipu_run_config

  @property
  def master(self):
    return self._master

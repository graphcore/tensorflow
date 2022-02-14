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
from tensorflow.python.ipu.utils import IPUConfig, IpuOptions
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation


class IPURunConfig(
    collections.namedtuple('IPURunConfig', [
        'iterations_per_loop', 'ipu_options', 'num_replicas', 'num_shards',
        'ordinal', 'prefetch_depth'
    ])):
  """IPU related configuration required by `IPUEstimator`.

  """
  def __new__(cls,
              iterations_per_loop=1,
              ipu_options=None,
              num_replicas=1,
              num_shards=1,
              ordinal=0,
              prefetch_depth=None):
    """ Creates an `IPURunConfig` instance.

    Args:
      iterations_per_loop: The number of mini-batches consumed on the IPU device
        before returning to the CPU host for each `Session.run`. The global step
        counter is increased by `iterations_per_loop` for every `Session.run`.
        The number of weight updates can be less than the number of iterations
        if gradient accumulation is used.
      ipu_options: An :py:class:`~tensorflow.python.ipu.utils.IPUConfig` which
        you have populated with your desired configuration options before
        creating this IPURunConfig. The `IPUEstimator` will then configure the
        IPU system with this `ipu_options` object when it builds your model.
      num_replicas: Number of replicated graphs (data parallelism)
      num_shards: Number of IPU devices on which the graph is sharded (model
        parallelism)
      ordinal: The IPU device ordinal to use.  For instance, 0 corresponds
        to `/device:IPU:0`.
      prefetch_depth: Integer or `None`. The `prefetch_depth` to be used by the
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` that is
        created internally.
    """
    num_devices = num_replicas * num_shards
    if num_devices > 1 and ipu_options is None:
      raise ValueError("IPU configuration requires more than one device, but"
                       " `ipu_options` is None")

    # Temporarily convert IPUConfig to IpuOptions pb to check device count.
    num_configured_devices = 1
    if ipu_options is not None:
      if not isinstance(ipu_options, IPUConfig):
        raise TypeError("ipu_options must be an IPUConfig instance.")
      pb = ipu_options._create_protobuf()  # pylint: disable=protected-access

      if ordinal >= len(pb.device_config):
        raise ValueError('Only {} device(s) available to choose from.'
                         ' You tried to pick a device at ordinal {}'.format(
                             len(pb.device_config), ordinal))
      if pb.device_config[ordinal].auto_count:
        num_configured_devices = pb.device_config[ordinal].auto_count
      else:
        # We're using cfg_index. Set equal for now and check later once the
        # device has been created
        num_configured_devices = num_devices

    if num_devices != num_configured_devices:
      raise ValueError(
          f"`IPURunConfig` configured with {num_devices} devices"
          f" ({num_replicas} replicas times {num_shards} shards), but"
          f" `ipu_options` configured with {num_configured_devices} devices")

    return super(IPURunConfig,
                 cls).__new__(cls,
                              iterations_per_loop=iterations_per_loop,
                              ipu_options=ipu_options,
                              num_replicas=num_replicas,
                              num_shards=num_shards,
                              ordinal=ordinal,
                              prefetch_depth=prefetch_depth)


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

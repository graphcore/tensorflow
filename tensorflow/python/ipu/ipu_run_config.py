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
    collections.namedtuple(
        'IPURunConfig',
        ['iterations_per_loop', 'ipu_options', 'compile_summary'])):
  r"""IPU related configuration required by `IPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps running in IPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
      Note that in evaluation don't use this value, instead we run total eval
      `steps` on IPU for a single `Session.run`.
    IpuOptions: An IpuOptions configuration protobuf which is populated prior
      to being passed into IPURunConfig
  """

  def __new__(cls,
              iterations_per_loop=2,
              ipu_options=None,
              compile_summary=False):
    return super(IPURunConfig,
                 cls).__new__(cls,
                              iterations_per_loop=iterations_per_loop,
                              ipu_options=ipu_options,
                              compile_summary=compile_summary)


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

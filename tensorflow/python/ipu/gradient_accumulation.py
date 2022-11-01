# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
Helper classes and methods for gradient accumulation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from enum import Enum


class GradientAccumulationReductionMethod(Enum):
  """Reduction method to use when accumulating gradients. We perform
  `gradient_accumulation_count` iterations (forward & backward passes)
  in each optimizer step, at the end of which we update the optimizer with
  gradients accumulated during the optimizer step. For each iteration within the
  optimizer step, the computed gradients can either be directly summed up or
  scaled such that we compute a mean of all gradients for each variable.
  Computing a mean avoids potential issues with overflow during accumulation,
  especially when using float16, but gives smaller gradients and might require
  adjusting the learning-rate accordingly.

  Note: The term `gradient_accumulation_count` is from the pipeline API
  and is referred to as `num_mini_batches` in
  :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`
  and
  :class:`~tensorflow.python.ipu.optimizers.CrossReplicaGradientAccumulationOptimizerV2`  # pylint: disable=line-too-long

  * SUM: Performs a sum of gradients
  * MEAN: Performs a sum of gradients scaled by (`1/num_mini_batches`)
  * RUNNING_MEAN: Performs a running mean of gradients
    (`acc*n/(n+1) + grad/(n+1)` for the nth iteration)
  """
  SUM = 0
  MEAN = 1
  RUNNING_MEAN = 2

  @classmethod
  def parse(cls, reduction_method):
    if isinstance(reduction_method, cls):
      return reduction_method

    if isinstance(reduction_method, str):
      key = reduction_method.upper()
      if key in cls.__members__:
        return cls[key]

    raise ValueError(f"Cannot parse {reduction_method} as one of "
                     "GradientAccumulationReductionMethod. Valid values are: "
                     f"{', '.join(cls._member_names_)}.")

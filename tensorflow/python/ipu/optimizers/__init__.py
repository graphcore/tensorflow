# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
Optimizer classes for the Graphcore IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=wildcard-import,unused-import

from tensorflow.python.ipu.optimizers.ipu_optimizer import IpuOptimizer
from tensorflow.python.ipu.optimizers.automatic_loss_scaling_optimizer import AutomaticLossScalingOptimizer
from tensorflow.python.ipu.optimizers.cross_replica_optimizer import CrossReplicaOptimizer
from tensorflow.python.ipu.optimizers.map_gradient_optimizer import MapGradientOptimizer
from tensorflow.python.ipu.optimizers.map_gradient_optimizer import _MapGradientOptimizerInvertedChaining
from tensorflow.python.ipu.optimizers.sharded_optimizer import ShardedOptimizer
from tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer import GradientAccumulationReductionMethod
from tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer import GradientAccumulationOptimizerV2
from tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer import CrossReplicaGradientAccumulationOptimizerV2
from tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer import GradientAccumulationOptimizer
from tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer import CrossReplicaGradientAccumulationOptimizer

# pylint: enable=wildcard-import,unused-import

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
Operations and utilities related to the Graphcore IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=wildcard-import,unused-import
from tensorflow.python.ipu.ops import all_to_all_op
from tensorflow.python.ipu.ops import all_to_all_op_grad
from tensorflow.python.ipu.ops import application_compile_op
from tensorflow.python.ipu.ops import control_flow_ops
from tensorflow.python.ipu.ops import control_flow_ops_grad
from tensorflow.python.ipu.ops import custom_ops
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu.ops import cross_replica_ops_grad
from tensorflow.python.ipu.ops import embedded_runtime
from tensorflow.python.ipu.ops import embedding_ops
from tensorflow.python.ipu.ops import embedding_ops_grad
from tensorflow.python.ipu.ops import functional_ops
from tensorflow.python.ipu.ops import functional_ops_grad
from tensorflow.python.ipu.ops import image_ops
from tensorflow.python.ipu.ops import internal_ops
from tensorflow.python.ipu.ops import internal_ops_grad
from tensorflow.python.ipu.ops import math_ops
from tensorflow.python.ipu.ops import nn_ops
from tensorflow.python.ipu.ops import nn_ops_grad
from tensorflow.python.ipu.ops import normalization_ops
from tensorflow.python.ipu.ops import normalization_ops_grad
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.ops import pipelining_ops_grad
from tensorflow.python.ipu.ops import rand_ops
from tensorflow.python.ipu.ops import rand_ops_grad
from tensorflow.python.ipu.ops import reduce_scatter_op
from tensorflow.python.ipu.ops import replication_ops
from tensorflow.python.ipu.ops import rnn_ops_grad
from tensorflow.python.ipu.ops import slicing_ops
from tensorflow.python.ipu.ops import slicing_ops_grad
from tensorflow.python.ipu.ops import statistics_ops
from tensorflow.python.ipu.ops import within_replica_ops
from tensorflow.python.ipu.ops import within_replica_ops_grad

from tensorflow.python.ipu import data
from tensorflow.python.ipu import config
from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_multi_worker_strategy
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import ipu_session_run_hooks
from tensorflow.python.ipu import keras
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import serving
from tensorflow.python.ipu import sharding
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_pipeline_estimator
from tensorflow.python.ipu import vertex_edsl

from tensorflow.python.ipu.keras import layers

from tensorflow.python.ipu.ops.experimental import popfloat_cast_to_gfloat

from tensorflow.python.ipu.optimizers import cross_replica_optimizer
from tensorflow.python.ipu.optimizers import ipu_optimizer
from tensorflow.python.ipu.optimizers import map_gradient_optimizer
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer

# Expose functional_ops.function as ops.outlined_function
from tensorflow.python.ipu.ops.functional_ops import outlined_function

# pylint: enable=wildcard-import,unused-import

sharding.enable_sharded_gradient_tape()

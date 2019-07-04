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
"""Operations and utilies related to the Graphcore IPU
"""

# pylint: disable=wildcard-import,unused-import
from tensorflow.python.ipu import autoshard
from tensorflow.python.ipu import autoshard_cnn
from tensorflow.python.ipu import gradient_accumulation_optimizer
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_optimizer
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import sharded_optimizer
from tensorflow.python.ipu import sharding
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import ipu_estimator

from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu.ops import embedding_ops
from tensorflow.python.ipu.ops import internal_ops
from tensorflow.python.ipu.ops import normalization_ops
from tensorflow.python.ipu.ops import rand_ops
from tensorflow.python.ipu.ops import rnn_ops
from tensorflow.python.ipu.ops import summary_ops
# pylint: enable=wildcard-import,unused-import
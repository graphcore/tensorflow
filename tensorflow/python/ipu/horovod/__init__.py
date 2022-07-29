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
from tensorflow.python.ipu.distributed import init, shutdown, size, local_size, rank, local_rank, mpi_threads_supported, mpi_enabled, mpi_built, gloo_enabled, gloo_built, nccl_built, ddl_built, is_homogeneous, Average, Sum, Adasum, allreduce, allgather, broadcast, gen_horovod_ops
from tensorflow.python.platform import tf_logging as logging

from .basics import *
from .ipu_horovod_strategy import *
from .popdist_strategy import *

logging.warning(
    'Module `%s` is deprecated in favour of `%s` and will be removed in a' \
    'future release.',
    'tensorflow.python.ipu.horovod', 'tensorflow.python.ipu.distributed')

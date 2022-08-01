# Copyright (C) 2019 Uber Technologies, Inc.
# Modifications copyright Microsoft
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
Horovod
~~~~~~~
"""
import atexit
import ctypes
import os
import re
from tensorflow.python.eager import context
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.ipu.distributed.basics import HorovodBasics
from tensorflow.python.ipu.distributed.gen_horovod_ops import (
    horovod_allreduce, horovod_broadcast, horovod_allgather)
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

# Loading the library will both register the ops and kernels and provide
# the "basic" C API through this wrapper.
_library_path = os.path.join(os.path.dirname(__file__), "horovod_plugin.so")
_basics = HorovodBasics(_library_path)

# Import the basic methods.
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported
mpi_enabled = _basics.mpi_enabled
mpi_built = _basics.mpi_built
gloo_enabled = _basics.gloo_enabled
gloo_built = _basics.gloo_built
nccl_built = _basics.nccl_built
ddl_built = _basics.ddl_built
is_homogeneous = _basics.is_homogeneous

# Import the reduction op values.
Average = _basics.Average
Sum = _basics.Sum
Adasum = _basics.Adasum


def _normalize_name(name):
  """Normalizes operation name to TensorFlow rules."""
  return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None, op=Sum):
  """An op which reduces an input tensor over all the Horovod processes. The
  default reduction is a sum.

  The reduction operation is keyed by the name of the op. The tensor type and
  shape must be the same on all Horovod processes for a given name. The
  reduction will not start until all processes are ready to send and receive
  the tensor.

  Returns:
    A tensor of the same shape and type as `tensor`, summed across all
    processes.
  """
  if name is None and not context.executing_eagerly():
    name = 'HorovodAllreduce_%s' % _normalize_name(tensor.name)
  return horovod_allreduce(tensor, name=name, reduce_op=op)


def allreduce(tensor, op=None):
  """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

  This function performs a bandwidth-optimal ring allreduce on the input
  tensor. If the input is an tf.IndexedSlices, the function instead does an
  allgather on the values and the indices, effectively doing an allreduce on
  the represented tensor.

  Arguments:
      tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
              The shape of the input must be identical across all ranks.
      op: The reduction operation to combine tensors across different ranks.
          Defaults to Average if None is given.

  Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
  """
  if op == Adasum:
    raise NotImplementedError('The Adasum reduction is not implemented.')

  if op is None:
    op = Average

  # Averaging happens in framework code, so translate that to Sum for the actual call
  true_op = Sum if op == Average else op

  if isinstance(tensor, indexed_slices.IndexedSlices):
    # For IndexedSlices, do two allgathers instead of an allreduce.
    horovod_size = math_ops.cast(size(), tensor.values.dtype)
    values = allgather(tensor.values)
    indices = allgather(tensor.indices)

    # To make this operation into an average, divide allgathered values by
    # the Horovod size.
    new_values = (values / horovod_size) if op == Average else values
    return indexed_slices.IndexedSlices(new_values,
                                        indices,
                                        dense_shape=tensor.dense_shape)

  horovod_size = math_ops.cast(size(), dtype=tensor.dtype)
  summed_tensor = _allreduce(tensor, op=true_op)
  new_tensor = (summed_tensor /
                horovod_size) if op == Average else summed_tensor
  return new_tensor


def allgather(tensor, name=None):
  """An op which concatenates the input tensor with the same input tensor on
  all other Horovod processes.

  The concatenation is done on the first dimension, so the input tensors on the
  different processes must have the same rank and shape, except for the first
  dimension, which is allowed to be different.

  Returns:
    A tensor of the same type as `tensor`, concatenated on dimension zero
    across all processes. The shape is identical to the input shape, except for
    the first dimension, which may be greater and is the sum of all first
    dimensions of the tensors in different Horovod processes.
  """
  if name is None and not context.executing_eagerly():
    name = 'HorovodAllgather_%s' % _normalize_name(tensor.name)
  return horovod_allgather(tensor, name=name)


def broadcast(tensor, root_rank, name=None):
  """An op which broadcasts the input tensor on root rank to the same input
  tensor on all other Horovod processes.

  The broadcast operation is keyed by the name of the op. The tensor type and
  shape must be the same on all Horovod processes for a given name. The
  broadcast will not start until all processes are ready to send and receive
  the tensor.

  Returns:
    A tensor of the same shape and type as `tensor`, with the value broadcasted
    from root rank.
  """
  if name is None and not context.executing_eagerly():
    name = 'HorovodBroadcast_%s' % _normalize_name(tensor.name)
  return horovod_broadcast(tensor, name=name, root_rank=root_rank)

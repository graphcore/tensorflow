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
# ==============================================================================
import re

from tensorflow.python.eager import context
from tensorflow.compiler.plugin.poplar.ops import gen_popdist_ops


def _normalize_name(name):
  """Normalizes operation name to TensorFlow rules."""
  return re.sub('[^a-zA-Z0-9_]', '_', name)


def allgather(value, tensor_name=None):
  if not tensor_name and not context.executing_eagerly():
    tensor_name = "PopDistAllGather_{}".format(_normalize_name(value.name))
  else:
    tensor_name = "Default"

  return gen_popdist_ops.popdist_all_gather(value, tensor_name=tensor_name)


def allreduce(value, reduce_op, tensor_name=None):
  if not tensor_name and not context.executing_eagerly():
    tensor_name = "PopDistAllReduce_{}".format(_normalize_name(value.name))
  else:
    tensor_name = "Default"

  return gen_popdist_ops.popdist_all_reduce(value,
                                            reduce_op=reduce_op.value,
                                            tensor_name=tensor_name)


def broadcast(value, root_rank=0, tensor_name=None):
  if not tensor_name and not context.executing_eagerly():
    tensor_name = "PopDistBroadcast_{}".format(_normalize_name(value.name))
  else:
    tensor_name = "Default"

  return gen_popdist_ops.popdist_broadcast(value, tensor_name=tensor_name)

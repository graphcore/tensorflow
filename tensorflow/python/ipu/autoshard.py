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
Automatic graph sharding
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ipu import autoshard_cnn
from tensorflow.python.ipu import sharding
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import deprecation


@deprecation.deprecated(None, "Use alternative execution modes, such as "
                        "pipelining, instead.")
def automatic_sharding(num_shards,
                       input_ts,
                       loss_ts,
                       edge_filter=None,
                       frozen_inference=False):
  """Automatically set shards for all connected nodes in graph.

  Args:
    num_shards: number of shards to split graph over.
    input_ts: tensor closest to the datafeed in graph.
    loss_ts: tensor closest to the loss in graph.
    edge_filter: a callable predicate, with the signature fn(edge), where edge
      is a tuple containing the name of the source op and the name of the
      destination op. If the predicate returns True then the graph will not be
      split at that edge. Only used if frozen_inference is False.
    frozen_inference: Flag set to True if running inference on a frozen graph.

  """
  autoshard_cnn.automatic_sharding(num_shards, input_ts, loss_ts, edge_filter,
                                   frozen_inference)


@tf_contextlib.contextmanager
def ipu_autoshard():
  """Provides a context for autosharding.  All operations created within this
  context will be automatically sharded.
  """

  attrs = {"_IpuAutoshard": attr_value_pb2.AttrValue(s=b"ON")}

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

  for op in ops.get_default_graph().get_operations():
    if sharding.has_attr(op, "_IpuAutoshard"):
      ops.get_default_graph().add_to_collection(sharding._IPU_AUTOSHARD, op)

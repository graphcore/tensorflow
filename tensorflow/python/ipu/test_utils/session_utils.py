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

import contextlib

from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib


@contextlib.contextmanager
def ipu_session(disable_grappler_optimizers=None):
  config = None
  # Disable any requested grappler optimizers
  if disable_grappler_optimizers and \
      isinstance(disable_grappler_optimizers, list):
    config = config_pb2.ConfigProto()
    for opt in disable_grappler_optimizers:
      assert hasattr(config.graph_options.rewrite_options, opt), \
          f"Tried to disable grappler optimizer '{opt}' but it's not an" \
          " attribute of the RewriterConfig proto"
      setattr(config.graph_options.rewrite_options, opt,
              rewriter_config_pb2.RewriterConfig.OFF)
  with session_lib.Session(config=config) as sess:
    yield sess


def move_variable_initialization_to_cpu():
  graph = ops.get_default_graph()

  init_ops = []
  dep_ops = [
      x.initializer.inputs[1].op for x in graph.get_collection('variables')
  ]
  visited = set()

  while dep_ops:
    op = dep_ops.pop()
    if not op in visited:
      visited.add(op)
      init_ops += [op]
      dep_ops += [x.op for x in op.inputs]

  # pylint: disable=protected-access
  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr(
        '_class',
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
            s=[b'loc:@cpu'])))
    op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
    op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
  # pylint: enable=protected-access

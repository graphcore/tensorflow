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

import networkx as nx
import numpy as np

from tensorflow.python.ipu import sharding
from tensorflow.python.ipu.sharding_utils import assign_shard, \
    convert_inference_ops_to_nx, calculate_memory, tensor_memory_use, \
    children, find_all_subgraphs, group_subgraphs, is_splitting_edge, \
    is_splitting_node, logging_helper
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.python.platform import tf_logging as logging

prohibited_ops = frozenset(["NextIteration", "PopDatastreamInfeedDequeue"])


def convert_ops_to_nx(fwd_ops, bwd_ops=None):
  """Convert ops into networkx graph.

  Args:
    fwd_ops: Iterable of tf.ops in the fwd graph.
    bwd_ops: Iterable of tf.ops in the bwd graph.

  Returns: Inference graph as networkx graph object,
          with each node having a memory attribute.

  """
  grad_ops = [op for op in bwd_ops if 'gradients/' in op.name.lower()]
  bwd_inputs = [t for op in grad_ops for t in op.inputs]
  graph = nx.DiGraph()
  dictionary = dict()
  # collect all variables including momentum types
  variable_ops = [
      op for op in fwd_ops + bwd_ops
      if op.type == 'ReadVariableOp' and op.inputs[0].op.type == 'VarHandleOp'
  ]
  var_mem = dict()
  all_variables = [t.name for t in trainable_variables()]
  for var in all_variables:
    # assign all memory for momentum type variables to root trainable variable
    var_ops = [
        op for op in variable_ops
        if op.inputs[0].name.startswith(var.split(":")[0])
    ]
    var_mem[var] = np.sum(
        [tensor_memory_use(t) for op in var_ops for t in op.outputs])
  variables_seen = []
  for op in fwd_ops:
    if op.type == 'ReadVariableOp' \
        and op.inputs[0].op.type == 'VarHandleOp' \
        and op.inputs[0].name not in variables_seen \
        and op.inputs[0].name in all_variables:
      parameter_mem = var_mem[op.inputs[0].name]
      variables_seen.append(op.inputs[0].name)
    else:
      parameter_mem = 0
    for t in op.outputs:
      if t in bwd_inputs and str(t.shape) != "<unknown>":
        logging.warning("Tensor '%s' has unknown shape." % t.name)
    bwd_links = [
        t for t in op.outputs
        if t in bwd_inputs and str(t.shape) != "<unknown>"
    ]
    if bwd_links != [] and op.type != 'ReadVariableOp' and not (
        op.type == 'Cast' and list(op.inputs)[0].op.type == 'ReadVariableOp'):
      saved_mem = np.sum([tensor_memory_use(t) for t in bwd_links])
    else:
      saved_mem = 0
    bwd_links = {
        t.name: {
            'size': tensor_memory_use(t),
            'shape': t.shape.as_list()
        }
        for t in bwd_links
    }
    has_bwd_links = bwd_links != {}
    graph.add_node(op.name,
                   bwd_links=bwd_links,
                   saved_mem=saved_mem,
                   has_bwd_links=has_bwd_links,
                   parameter_mem=parameter_mem)
    dictionary[op.name] = op

  for op in fwd_ops:
    for c_op in children(op):
      if c_op in fwd_ops:
        graph.add_edges_from([(op.name, c_op.name)])

  return graph


def automatic_sharding(num_shards,
                       input_ts,
                       output_ts,
                       edge_filter=None,
                       frozen_inference=False):
  """Automatically set shards for all connected nodes in graph.

  Args:
      num_shards(int): number of shards to split graph over.
      input_ts(tf.Tensor): tensor closest to the data-feed in graph.
      output_ts(tf.Tensor): tensor closest to the output/loss in graph.
      edge_filter: a callable predicate, with the signature fn(edge), where
                   edge is a tuple containing the name of the source op and
                   the name of the destination op. If the predicate returns True
                   then the graph will not be split at that edge.
                   Only used if frozen_inference is False.
      frozen_inference: Flag set to True if running inference on a frozen graph.

  Raises: ValueError if no ops are set to run on IPU device.

  """

  output_op = output_ts.op
  input_op = input_ts.op

  ipu_ops = list(
      filter(lambda o: 'IPU' in o.device, output_op.graph.get_operations()))
  if len(ipu_ops) == 0:
    raise ValueError("No ops placed on IPU device to shard.")

  fwd_ops = []
  marked_collection = output_op.graph.get_collection(sharding._IPU_AUTOSHARD)
  if len(marked_collection) > 0:
    fwd_ops = marked_collection
  else:
    for op in ipu_ops:
      if not any([s in op.name.lower() for s in ['gradients/', '/update_']]):
        fwd_ops.append(op)
  bwd_ops = [o for o in ipu_ops if o not in fwd_ops]
  fwd_ops = [o for o in fwd_ops if o.type not in prohibited_ops]

  if input_op not in fwd_ops:
    input_op = [op for op in input_ts.consumers() if op in fwd_ops][0]

  if frozen_inference:
    graph = convert_inference_ops_to_nx(fwd_ops)
  else:
    graph = convert_ops_to_nx(fwd_ops, bwd_ops)

  # Check graph is a single weakly connected component
  # if not find the component with the output op in and use that
  weakly_connected_components = list(nx.weakly_connected_components(graph))
  graph_fwd = None
  for g in weakly_connected_components:
    if output_op.name in g:
      graph_fwd = graph.subgraph(g)
      break
  fwd_ops = [op for op in fwd_ops if op.name in graph_fwd.nodes]

  if nx.number_weakly_connected_components(graph_fwd) != 1:
    raise RuntimeError(
        "Error: number of disconnected subgraphs in auto-sharder is {}".format(
            nx.number_weakly_connected_components(graph)))
  splitting_edges = []
  if frozen_inference:
    # Find all graph ops that when split at their output can create two sub-graphs
    # where the input and output are not in the same sub-graph
    for node in graph_fwd.nodes:
      if is_splitting_node(graph_fwd, node, input_op.name, output_op.name):
        splitting_edges.append([(node, v) for v in graph_fwd.successors(node)])
  else:
    # Find all graph edges that split the graph into two subgraphs where the input
    # and output are not in the same subgraph
    for edge in graph_fwd.edges:
      if is_splitting_edge(graph_fwd, edge, input_op.name, output_op.name):
        splitting_edges.append([edge])

    if edge_filter and callable(edge_filter):
      splitting_edges = list(
          filter(lambda e: not edge_filter(e[0]), splitting_edges))
  logging.debug('Possible splitting edges ' + str(splitting_edges))

  # Verify that we have enough subgraphs to fill all of the available shards
  if len(splitting_edges) + 1 < num_shards:
    raise Exception(
        "There are fewer subgraphs (%s) than available shards (%s). Reduce the "
        "number of shards." % (len(splitting_edges) + 1, num_shards))

  # Given the splitting edges found find all of the subgraphs created and order them
  sub_graphs = find_all_subgraphs(graph_fwd, splitting_edges, input_op.name,
                                  output_op.name, [op.name for op in fwd_ops])
  sub_graph_mem = [calculate_memory(g) for g in sub_graphs]
  logging_helper(sub_graphs)
  best_ind = group_subgraphs(sub_graph_mem, num_shards)
  logging.debug('Splitting edges ' +
                str(list(map(lambda x: str(splitting_edges[x]), best_ind))))

  ind_pad = [0] + [i + 1 for i in best_ind] + [len(sub_graph_mem)]
  per_shard_sub_graphs = []
  for i in range(num_shards):
    per_shard_sub_graphs.append(
        graph_fwd.subgraph([
            nodes for g in sub_graphs[ind_pad[i]:ind_pad[i + 1]]
            for nodes in g.nodes
        ]))
  logging_helper(per_shard_sub_graphs)
  assign_shard(fwd_ops, ipu_ops, per_shard_sub_graphs)

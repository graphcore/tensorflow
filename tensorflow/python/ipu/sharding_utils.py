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
import itertools

import networkx as nx
from networkx.algorithms.components.weakly_connected import _plain_bfs as bfs
import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.python.ipu import sharding
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.platform import tf_logging as logging

prohibited_ops = frozenset(["NextIteration", "PopDatastreamInfeedDequeue"])


def ordered_weakly_connected_components(graph, fwd_nodes):
  """Generate weakly connected components of graph.

    Args:
      graph : NetworkX directed graph
      fwd_nodes: Collection of nodes as they appear in the directed graph from input to output

    Returns: Generator of sets
          A generator of sets of nodes, one for each weakly connected
          component of graph, preserving order based on fwd_nodes
  """
  seen = set()
  for node in graph.nbunch_iter(fwd_nodes):
    if node not in seen:
      c = set(bfs(graph, node))
      yield c
      seen.update(c)


def convert_inference_ops_to_nx(fwd_ops):
  """Convert inference ops into networkx graph.

  Args:
    fwd_ops: Iterable of tf.ops in the inference graph.

  Returns: Inference graph as networkx graph object,
          with each node having a memory attribute.

  """
  graph = nx.DiGraph()

  # Add nodes to the graph
  for op in fwd_ops:
    activation_mem = 0
    parameter_mem = 0
    if op.type == 'Const':
      parameter_mem = tensor_memory_use(op.outputs[0])
      activation_mem = 0
    elif op.type not in ('NoOp',):
      activation_mem = tensor_memory_use(op.outputs[0])
      parameter_mem = 0

    graph.add_node(op.name,
                   saved_mem=activation_mem,
                   parameter_mem=parameter_mem)

  # Add edges to the graph.
  for op in fwd_ops:
    for c_op in children(op):
      if c_op in fwd_ops:
        graph.add_edges_from([(op.name, c_op.name)])

  return graph


def tensor_memory_use(tensor):
  """Memory use of a tensor in bytes.
  This is a model for the size of a tensor on a tile.  It could be improved by
  accepting that tensors generate vertex state associated with the number of
  operations which consume them.

  Args:
    tensor(tf.Tensor): Input tensor.

  Returns:Number of bytes the tensor consumes.

  """
  return tensor.shape.num_elements() * tensor.dtype.size


def children(input_op):
  """Find successor ops for input_op.

  Args:
    input_op(tf.Operation): Input op.

  Returns: Collection of successors with a direct edge from the input_op.

  """
  return set(child_op for out in input_op.outputs
             for child_op in out.consumers())


def set_ipu_shard(op, index):
  """Set shard index for op.

  Args:
    op(tf.Operation): Operator
    index(int): IPU index

  """
  proto = xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MAXIMAL,
                                  tile_assignment_devices=[index])

  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  op._set_attr(sharding._XLA_SHARDING, attr_value)


def is_splitting_edge(g_fwd, edge, input_node, output_node):
  """Check if `edge` splits `g_fwd` into two weakly connected graphs,
  with `input_node` and `output_node` in different sub-graphs.

  Args:
    g_fwd(nx.DiGraph): Input graph.
    edge(Tuple): Splitting edge.
    input_node(networkx.classes.reportview.NodeView): Input node for the entire graph.
    output_node(networkx.classes.reportview.NodeView): Output node for the entire graph.

  Returns: True if edge is a splitting edge.

  """
  g_copy = nx.DiGraph(g_fwd)
  g_copy.remove_edge(edge[0], edge[1])
  weakly_connected_comps = list(nx.weakly_connected_components(g_copy))
  if len(weakly_connected_comps) == 2:
    if not any([(input_node in c) and (output_node in c)
                for c in weakly_connected_comps]):
      return True
  return False


def is_splitting_node(g_fwd, node, input_node, output_node):
  """Check if `g_fwd` can be split into two weakly connected graphs at the output of node,
  with input_node and output_node in different sub-graphs.

  This is useful when the graph has skip connections, i.e. same node connected to more than one successors.
  Using the  previous `is_splitting_edge` logic, makes such nodes in-eligible to split,
  leaving a large sub-graph in one shard.

  Args:
    g_fwd(nx.DiGraph): Input graph.
    node(networkx.classes.reportview.NodeView): Splitting node.
    input_node(networkx.classes.reportview.NodeView): Input node for the entire graph.
    output_node(networkx.classes.reportview.NodeView): Output node for the entire graph.

  Returns: True if `g_fwd` can be split at the output of `node`.

  """
  g_fwd = nx.DiGraph(g_fwd)
  edges_out = [(node, v) for v in g_fwd.successors(node)]
  g_fwd.remove_edges_from(edges_out)
  weakly_connected = list(nx.weakly_connected_components(g_fwd))
  if len(weakly_connected) == 2:
    if not any([(input_node in c) and (output_node in c)
                for c in weakly_connected]):
      return True
  return False


def calculate_memory(graph, parameter=True, saved=True, peak_activation=False):
  """Calculate memory consumption of the graph based on flags.

  Args:
    graph(nx.DiGraph): Input graph with attributes for memory usage.
    parameter(bool): If True, only count parameter memory.
    saved(bool): If True, only count activation memory.
    peak_activation(bool): If True, only count peak activation memory.

  Returns: Memory consumed by the graph.

  """
  total_mem = 0
  peak_activation_mem = 0
  parameter_mem = 0

  for node in graph.nodes:
    total_mem += graph.nodes[node]['saved_mem']
    total_mem += graph.nodes[node]['parameter_mem']

    peak_activation_mem = max(peak_activation_mem,
                              graph.nodes[node]['saved_mem'])

    parameter_mem += graph.nodes[node]['parameter_mem']

  total = parameter and saved
  if peak_activation:
    return peak_activation_mem
  if total:
    return total_mem
  if parameter:
    return parameter_mem
  if saved:
    return total_mem - parameter_mem


def find_all_subgraphs(graph, splitting_edges, input_node, output_node,
                       fwd_nodes):
  """Split graph into subgraphs based on splitting edges.

  Args:
    graph(nx.DiGraph): Input graph.
    splitting_edges(Collection of lists): Nested collection of splitting edges.
    input_node(networkx.classes.reportview.NodeView): Input node for the entire graph.
    output_node(networkx.classes.reportview.NodeView): Output node for the entire graph.
    fwd_nodes: Collection of nodes in the graph from input to output order.

  Returns(List of nx.DiGraph): sub-graphs created by the splitting_edges.

  """
  graph = nx.DiGraph(graph)
  for edges in splitting_edges:
    graph.remove_edges_from(edges)
  subgraphs = [
      graph.subgraph(c)
      for c in ordered_weakly_connected_components(graph, fwd_nodes)
  ]
  if input_node not in subgraphs[0]:
    raise RuntimeError("input node must be in first sub-graph")

  if output_node not in subgraphs[-1]:
    raise RuntimeError("output node must be in final sub-graph")

  return subgraphs


def group_subgraphs(subgraph_mem, num_shards):
  """Split the ordered subgraphs into n groups and calculate the memory for each possible combination.

  Choose the best grouping based on:
        1. minimum peak memory
        2. variance of memory
  (TODO): could use minimum data transferred between IPUs?

  Args:
    subgraph_mem(List): List of memory consumption per sub-graph.
    num_shards(int): Number of shards to group the sub-graphs into.

  Returns(List[List]): Collection of num_shards elements, each holding indices of subgraphs per shard.

  """
  min_peak_mem = np.inf
  best_ind = []
  best_mem = []

  # Calculate memory usage for all possible sorted combinations of subgraphs.
  for ind in itertools.combinations(range(len(subgraph_mem)), num_shards - 1):
    # Padding ensures that the first sub-graph and last sub-graph include the input and the output respectively.
    ind_pad = [0] + [i + 1 for i in ind] + [len(subgraph_mem)]
    mem = []
    for i in range(num_shards):
      mem.append(np.sum(subgraph_mem[ind_pad[i]:ind_pad[i + 1]]))

    max_mem = np.max(mem)
    if max_mem < min_peak_mem:
      best_ind = [ind]
      best_mem = [mem]
      min_peak_mem = max_mem
    elif max_mem == min_peak_mem:
      best_ind += [ind]
      best_mem += [mem]

  # Use mem variance to break ties.
  min_var = np.inf
  for ind, mem in zip(best_ind, best_mem):
    var_mem = np.var(mem)
    if var_mem < min_var:
      best_ind = [ind]
      best_mem = [mem]
      min_var = var_mem
    elif var_mem == min_var:
      best_ind += [ind]
      best_mem += [mem]

  # If still tied choose the first option in the list
  return best_ind[0]


def logging_helper(per_shard_subgraphs):
  """Helper to log memory use per subgraph."""

  logging.debug('Per shard sub-graph memory use ' + str([
      "{:.4g} MiB".format(float(calculate_memory(g)) / (1024 * 1024))
      for g in per_shard_subgraphs
  ]))

  logging.debug('Per shard sub-graph memory use (variables only) ' + str([
      "{:.4g} MiB".format(
          float(calculate_memory(g, saved=False)) / (1024 * 1024))
      for g in per_shard_subgraphs
  ]))

  logging.debug('Per shard sub-graph memory use (activations only) ' + str([
      "{:.4g} MiB".format(
          float(calculate_memory(g, parameter=False)) / (1024 * 1024))
      for g in per_shard_subgraphs
  ]))


def assign_shard(fwd_ops, ipu_ops, per_shard_subgraphs):
  """Assign shard index for every op in fwd_ops and ipu_ops.

  Args:
    fwd_ops(List[tf.Operation]): Collection of ops in the sub-graph where loss op exists.
    ipu_ops(List[tf.Operation]): Collection of all ipu_ops.
    per_shard_subgraphs: Collection of nx.DiGraphs each nx.DiGraph holding ops per shard.

  """
  for op in fwd_ops:
    shard_set = False
    for i, g in enumerate(per_shard_subgraphs):
      if op.name in g:
        set_ipu_shard(op, i)
        shard_set = True
    if not shard_set:
      raise RuntimeError("%s not in any graph split" % op.name)

  for op in filter(lambda o: o not in fwd_ops, ipu_ops):
    attr = sharding.get_shard_from_colocation(op)
    if not attr:
      for child in children(op):
        attr = sharding.get_shard_from_colocation(child)
    if attr:
      op._set_attr(sharding._XLA_SHARDING, attr_value_pb2.AttrValue(s=attr))

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
Outfeed queue
~~~~~~~~~~~~~
"""

from enum import Enum

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation


class IPUOutfeedMode(Enum):
  """Types used to control the IPUOutfeedQueue modes.

  Contains the following values:

  * `ALL` - When used with an IPUOutfeedQueue, all the elements which were
    enqueued to the queue will be returned by the outfeed.
  * `LAST` - When used with an IPUOutfeedQueue, only the last element which was
    enqueued to the queue will be returned by the outfeed.

  """
  ALL = "all"
  LAST = "get_last"


class IPUOutfeedQueue:
  """Generates and adds outfeed enqueue/dequeue operations to the graph.

  An outfeed is the counterpart to an infeed and manages the
  transfer of data (like tensors, tuples or dictionaries of tensors)
  from the IPU graph to the host.

  The queue has two modes of operation - outfeed all or outfeed last.
  In outfeed all mode every element that is enqueued will be stored
  for a subsequent dequeue. All of the enqueued elements will be returned
  when the dequeue operation is run. This is the default behaviour.

  In outfeed last mode only the last enqueued element is stored. The dequeue
  operation will in this case return a single element.
  """
  @deprecation.deprecated_args(None, "Use outfeed_mode instead.",
                               "outfeed_all")
  def __init__(self,
               feed_name,
               outfeed_mode=None,
               outfeed_all=None,
               device_ordinal=0,
               replication_factor=1,
               io_batch_size=1):
    """Creates an IPUOutfeedQueue object.

    Args:
        feed_name: a user provided name for the outfeed operation. Must be
          unique within all IPUOutfeedQueue and IPUInfeedQueue
          operations.
        outfeed_mode: `ipu_outfeed_queue.IPUOutfeedMode` type used to control the
          outfeed behaviour. If not specified then all elements will be
          returned by the outfeed when the dequeue operation is run.
        outfeed_all: deprecated.
        device_ordinal: ordinal of the IPU device on which this queue will be
          used. By default the queue will be used on "/device/IPU:0".
        replication_factor: the number of replicated graphs this Outfeed
          will be used in.
        io_batch_size: Output tensors will be batched into this number of samples
          before being sent to the host.  This reduces the amount of
          device->host communication at the expense of needing to store the
          tensors on the device, and the extra computation required to operate
          the batching.

    Raises:
      ValueError: if the types or values are incorrect
      """

    # Handle deprecated factor.
    if outfeed_all is not None:
      logging.warning("`outfeed_all` has been deprecated and will be removed "
                      "in the future version. Use `outfeed_mode` instead.")

      if not isinstance(outfeed_all, bool):
        raise ValueError("Expcted value True or False for outfeed_all")

      outfeed_mode = IPUOutfeedMode.ALL if outfeed_all else IPUOutfeedMode.LAST

    # Default to all.
    self._outfeed_mode = outfeed_mode or IPUOutfeedMode.ALL

    if not isinstance(self._outfeed_mode, IPUOutfeedMode):
      raise ValueError("Expcted `outfeed_mode` value to be of "
                       "`ipu_outfeed_queue.IPUOutfeedMode` type, but is %s." %
                       (str(type(outfeed_mode))))

    if not isinstance(device_ordinal, int):
      raise ValueError('Device ordinal must be an integer')

    if device_ordinal < 0:
      raise ValueError('Device ordinal must be >= 0')

    if replication_factor < 1:
      raise ValueError('Replication factor must be >= 1')

    self._outfeed_all = self._outfeed_mode == IPUOutfeedMode.ALL
    self._device_ordinal = device_ordinal
    self._replication_factor = replication_factor
    self._io_batch_size = max(1, io_batch_size)
    self._feed_name = str(feed_name)

    self._operations = []
    self._structure = None
    self._device_str = '/device:IPU:{}'.format(str(device_ordinal))

  def enqueue(self, tensors):
    """Enqueue a tensor, tuple or a dictionary of tensors for being outfed
    from the IPU graph. This operation is placed on the IPU device.
    This function returns an Operation which needs be executed (by either
    returning it or using tf.control_dependencies(...))

    Examples:

    1. Outfeed returning a single tensor:

    .. code-block:: python

       outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

       def body(v):
         v = v + 1
         outfeed = outfeed_queue.enqueue(v)
         return (v, outfeed)

       def my_net(v):
         r = loops.repeat(20, body, (v))
         return r

       with ipu.scopes.ipu_scope("/device:IPU:0"):
         res = ipu_compiler.compile(my_net, inputs=[v])

       ...
       ...

    2. Outfeed returning a tuple of tensors:

    .. code-block:: python

       outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

       def body(v):
         v = v + 1
         x = v * 2
         outfeed = outfeed_queue.enqueue((v, x))
         return (v, outfeed)

       def my_net(v):
         r = loops.repeat(20, body, (v))
         return r

       with ipu.scopes.ipu_scope("/device:IPU:0"):
         res = ipu_compiler.compile(my_net, inputs=[v])

       ...
       ...

    3. Outfeed returning a dictionary of tensors:

    .. code-block:: python

       outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

       def body(v):
         v = v + 1
         x = v * 2
         outfeed = outfeed_queue.enqueue({"output_1": v,
                                          "output_2": x})
         return (v, outfeed)

       def my_net(v):
         r = loops.repeat(20, body, (v))
         return r

       with ipu.scopes.ipu_scope("/device:IPU:0"):
         res = ipu_compiler.compile(my_net, inputs=[v])

       ...
       ...

      """
    if self.enqueued:
      raise ValueError("An outfeed can only be enqueued once.")

    self._structure = _OutfeedStructure(tensors, self._replication_factor)
    with ops.device(self._device_str):
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          self._structure.to_tensor_list(tensors),
          output_shapes=self._structure.flat_shapes,
          outfeed_mode=self._outfeed_mode.value,
          feed_id=self._feed_name,
          replication_factor=self._replication_factor,
          io_batch_size=self._io_batch_size)

    self._operations.append(outfeed_op)
    return outfeed_op

  @property
  def enqueued(self):
    enqueued_graphs = set()

    def add_graphs(g):
      enqueued_graphs.add(g)
      if isinstance(g, func_graph.FuncGraph):
        # Consider all outer graphs enqueued as well.
        add_graphs(g.outer_graph)

    for o in self._operations:
      add_graphs(o.graph)

    current_graph = ops.get_default_graph()
    return current_graph in enqueued_graphs

  def dequeue(self):
    """Generate host side operation to dequeue the outfeed values. The
    operation generated by this function will block if called prior
    to any enqueues.

    The return value of this operation depends on the enqueued tensors,
    replication factor and the execution mode.

    Examples:

    1. Outfeed returning a single tensor:

    .. code-block:: python

        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed",
                                                          replication_factor=2)

        def body(input):
          output = input + 1
          outfeed = outfeed_queue.enqueue(output)
          return (output, outfeed)

        def my_net(input):
          r = loops.repeat(20, body, (input))
          return r

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          res = ipu_compiler.compile(my_net, inputs=[v])

        with ops.device('cpu'):
          v = tf.placeholder(np.float32, [4, 4])

        outfeed = outfeed_queue.dequeue()
        with tf.Session() as sess:
          result = sess.run(res, {v:np.ones([4, 4], np.float32)})
          outfed = sess.run(outfeed)

    In this example the tensor `output` is of shape [4, 4] and it's enqueued
    into the outfeed with replication_factor = 2. If the `outfeed_mode` is
    `outfeed_mode == IPUOutfeedMode.ALL`, then the shape of the resulting
    `outfed` tensor will be [20, 2, 4, 4], where the first dimension represents
    the number of times we have enqueued a tensor to the outfeed - in this
    example the loop is repeated 20 times, and therefore we get 20 values back
    from the outfeed. The second dimension is the replication_factor, which
    allows us to see the individual values from each replicated graph. If the
    `outfeed_mode` is `outfeed_mode == IPUOutfeedMode.LAST`, then the shape of
    the resulting `outfed` tensor will be [2, 4, 4], which represents the value
    of the output tensor the last time it was enqueued during execution for
    each of the replicated graphs.

    2. Outfeed returning a tuple of tensors:

    .. code-block:: python

        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

        def body(input):
          output = input + 1
          sum = tf.reduce_sum(output)
          outfeed = outfeed_queue.enqueue((output, sum))
          return (output, outfeed)

        def my_net(input):
          r = loops.repeat(20, body, (input))
          return r

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          res = ipu_compiler.compile(my_net, inputs=[v])

        with ops.device('cpu'):
          v = tf.placeholder(np.float32, [4, 4])

        outfeed = outfeed_queue.dequeue()
        with tf.Session() as sess:
          result = sess.run(res, {v:np.ones([4, 4], np.float32)})
          outfed = sess.run(outfeed)

    In this example we outfeed a tuple of tensors, `output` and `sum`, where
    the former is of shape [4, 4] and latter [1]. If the `outfeed_mode` is
    `outfeed_mode == IPUOutfeedMode.ALL`, then the resulting outfed is a
    two-tuple of tensors with shapes ([20, 4, 4], [20, 1]), where the first
    dimension in each of the tensors represents the number of times we have
    enqueued these tensors to the outfeed - in this example the loop is repeated
    20 times, and therefore we get 20 values back from the outfeed for each of
    the tensors in the tuple. If the `outfeed_mode` is
    `outfeed_mode == IPUOutfeedMode.LAST`, then the `outfed` is a two tuple of
    tensors with shapes ([4, 4], [1]), which represents the values of the
    `output` and `sum` tensors the last time they were enqueued during
    execution.

    Note that `replication_factor` here is the default (=1), which means that
    the extra replication dimension is not added.

    3. Outfeed returning a dictionary of tensors:

    .. code-block:: python

        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed",
                                                          replication_factor=8)

        def body(input):
          output = input + 1
          sum = tf.reduce_sum(output)
          outfeed = outfeed_queue.enqueue({"x": output,
                                           "y": sum})
          return (output, outfeed)

        def my_net(input):
          r = loops.repeat(40, body, (input))
          return r

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          res = ipu_compiler.compile(my_net, inputs=[v])

        with ops.device('cpu'):
          v = tf.placeholder(np.float32, [4, 4])

        outfeed = outfeed_queue.dequeue()
        with tf.Session() as sess:
          result = sess.run(res, {v:np.ones([4, 4], np.float32)})
          outfed = sess.run(outfeed)

    In this example we outfeed a dictionary of tensors, `output` and `sum`,
    where the former is of shape [4, 4] and latter [1]. If the `outfeed_mode` is
    `outfeed_mode == IPUOutfeedMode.ALL`, then the resulting outfed is a
    dictionary of tensors with shapes: {"x": [40, 8, 4, 4], "y": [40, 8, 1]},
    where the first dimension in each of the tensors represents the number of
    times we have enqueued these tensors to the outfeed - in this example the
    loop is repeated 40 times, and therefore we get 40 values back from the
    outfeed for each of the tensors in the tuple. The second dimension is the
    replication_factor, which allows us to see the individual values from each
    replicated graph. If the `outfeed_mode` is
    `outfeed_mode == IPUOutfeedMode.LAST`, then the `outfed` is a dictionary of
    tensors with shapes: {"x": [8, 4, 4], "y": [8, 1]}, which represents the
    values of the `output` and `sum` tensors the last time they were enqueued
    during execution for each of the replicated graphs.

    """
    if not self.enqueued:
      raise ValueError(
          "Trying to dequeue an outfeed which has not been enqueued.")
    with ops.device('cpu'):
      outfeed_dequeue = \
        gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
            output_types=self._structure.flat_types,
            output_shapes=self._structure.flat_shapes,
            outfeed_mode=self._outfeed_mode.value,
            feed_id=self._feed_name,
            device_ordinal=self._device_ordinal,
            replication_factor=self._replication_factor)

    return self._structure.from_tensor_list(outfeed_dequeue)

  @property
  def deleter(self):
    """A `tf.Operation` that can be run to delete the resources owned
    by this IPUOutfeedQueue. This allows creating a new IPUOutfeedQueue
    with the same name afterwards. The behaviour is undefined if this
    op is executed concurrently with the dequeue op.

    Returns:
      A `tf.Operation` that can be run to delete this IPUOutfeedQueue
    """
    return gen_pop_datastream_ops.ipu_delete_outfeed(
        feed_id=self._feed_name, device_ordinal=self._device_ordinal)


class _OutfeedStructure:
  """ An internal class used for storing the structure of the IPUOutfeedQueue.
  """
  def __init__(self, tensors, replication_factor):
    self._singular = False
    self._tuple = False
    self._list = False
    self._dict = False
    self._dict_keys = []

    flat_types = []
    flat_shapes = []
    # Create the data structure depending on the input type.
    if isinstance(tensors, ops.Tensor):
      self._singular = True
      flat_types = [tensors.dtype]
      flat_shapes = [tensors.get_shape()]

    elif isinstance(tensors, (tuple, list)):
      self._tuple = isinstance(tensors, tuple)
      self._list = isinstance(tensors, list)
      # We require all the elements to be tensors.
      if not self._check_list_of_all_type(ops.Tensor, tensors):
        raise ValueError("""\
Expected all values in the outfeed tuple to be TensorFlow tensors.""")
      for tensor in tensors:
        flat_types.append(tensor.dtype)
        flat_shapes.append(tensor.get_shape())

    elif isinstance(tensors, dict):
      self._dict = True
      # We require all the keys to be strings.
      if not self._check_list_of_all_type(str, tensors.keys()):
        raise ValueError("""\
Expected all keys in the outfeed dictionary to be strings.""")
      # We require all the values to be tensors.
      if not self._check_list_of_all_type(ops.Tensor, tensors.values()):
        raise ValueError("""\
Expected all values in the outfeed dictionary to be TensorFlow tensors.""")
      for key in tensors:
        tensor = tensors[key]
        self._dict_keys.append(key)
        flat_types.append(tensor.dtype)
        flat_shapes.append(tensor.get_shape())

    else:
      raise ValueError("""\
IPUOutfeedQueue Enqueue input needs to be either:
* TensorFlow tensor
* Tuple of TensorFlow tensors
* Dictionary of strings to TensorFlow tensors""")

    # We add an extra dimension when the replication factor is greater than 1.
    if replication_factor > 1:
      flat_shapes = [
          tensor_shape.TensorShape([replication_factor]).concatenate(shape)
          for shape in flat_shapes
      ]

    self._flat_structure = {
        "output_shapes": flat_shapes,
        "output_types": flat_types,
    }

  @staticmethod
  def _check_list_of_all_type(type, list):
    return all(isinstance(x, type) for x in list)

  @property
  def flat_structure(self):
    return self._flat_structure

  @property
  def flat_shapes(self):
    return self._flat_structure["output_shapes"]

  @property
  def flat_types(self):
    return self._flat_structure["output_types"]

  def to_tensor_list(self, tensors):
    if self._singular:
      return [tensors]
    if self._tuple or self._list:
      return list(tensors)
    if self._dict:
      return list(tensors.values())
    raise ValueError("Can't be reached")

  def from_tensor_list(self, flat_tensors):
    # We require the input to be a list of flat_tensors.
    if (not isinstance(flat_tensors, list)
        or not self._check_list_of_all_type(ops.Tensor, flat_tensors)):
      raise ValueError("""\
Expected flat_tensors to be a list of TensorFlow tensors.""")

    if self._singular:
      return flat_tensors[0]
    if self._tuple:
      return tuple(flat_tensors)
    if self._list:
      return flat_tensors
    if self._dict:
      return dict(zip(self._dict_keys, flat_tensors))
    raise ValueError("Can't be reached")

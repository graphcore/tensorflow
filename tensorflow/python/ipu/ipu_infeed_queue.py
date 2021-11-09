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
Infeed queue
~~~~~~~~~~~~
"""
import threading

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import loops
from tensorflow.python.util import nest
from tensorflow.python.framework import type_spec

_uid_counter = 0
_uid_lock = threading.Lock()
_internal_id = "_internal_id"


def _generate_unique_name():
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return str(uid)


class IPUInfeedQueue:
  """Wraps a tf.Dataset object with infeed operations specific to the IPU.

  This class, along with `tensorflow.python.ipu.loops` is used to create a data
  pipeline from a `dataset` into a training/inference loop on the IPU inside a
  single `session.run` which reduces the overheads of calling `session.run` for
  each iteration of the loop.

  You should pass the infeed queue as an argument to a loop from
  `tensorflow.python.ipu.loops`. These loops will then handle the dequeuing of
  the data to the device automatically.

  The following skeleton shows how to use this method when building a training
  loop. Note how the body signature contains variables which correspond to the
  nested structure of `tf.Tensor` objects representing the next element in the
  infeed queue:

  .. code-block:: python

    # Create an example dataset.
    dataset = ...  # A `tf.data.Dataset` object.

    def dataset_parser(value):
      features, labels = parse_record(value)
      return {"features": features,
              "labels": labels}
    # The resulting dataset has a nested structure of: {features, labels}.
    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    # dataset can no longer be used beyond this point.

    def my_net():
      # Note how the nested structure forms part of the loop body signature.
      def body(loss, features, labels):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = tf.conv2d(features, .....)
          ...
          ...
          logits = tf.nn.xw_plus_b(....)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        optimizer = gradient_descent.GradientDescentOptimizer(0.000001)
        train = optimizer.minimize(loss)
        with ops.control_dependencies([train]):
          return array_ops.identity(loss)

      loss = 0.0
      return = tf.python.ipu.loops.repeat(10000, body, [loss], infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    with tf.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      result = sess.run(res)

  """
  def __init__(
      self,
      dataset,
      device_ordinal=None,
      prefetch_depth=None,
      **kwargs):
    """Creates an IPUInfeedQueue object.

    Args:
      dataset: a `tf.data.Dataset` object, all transformations e.g. `shuffle`,
        `repeat`, `batch` must be applied prior to passing in to this function.
        This dataset can no longer be used after creating this queue.
      device_ordinal: Integer ordinal of the IPU device on which this queue will
        be used. If not specified will try and deduce the IPU device from the
        current strategy and if that fails will default to "/device:IPU:0".
      prefetch_depth: the number of elements Poplar will prefetch.
        The depth of the Poplar datastream buffer size which may be prefetched
        before being read by the device. By default the prefetch_depth size is
        automatically determined (currently defaults to 1). Increasing the size
        of the prefetch_depth allows for prefetching of multiple entries,
        increasing the probability there will be a valid entry in the buffer for
        the device to read before falling back to synchronously fetching the
        next entry. This value has to be greater than zero.

    Raises:
      ValueError: if all dimensions of shapes of dataset.output_shapes are not
        fully defined. tf.data.batch function must be called with
        `drop_remainder=True` to ensure that batch size is constant.

    """

    for output_shape in dataset._flat_structure["output_shapes"]:
      if isinstance(output_shape, list) or isinstance(output_shape, tuple):
        raise ValueError("Nested list/tuple input shapes are not supported")
      if not output_shape.is_fully_defined():
        raise ValueError("""Output shape {} is not fully defined. If using \
tf.Dataset.batch, set `drop_remainder=True`.""".format(output_shape))
    if prefetch_depth is None:
      prefetch_depth = 1
    if prefetch_depth <= 0:
      raise ValueError(
          "prefetch_depth must be greater than zero, but it is {}".format(
              prefetch_depth))
    if prefetch_depth > 255:
      raise ValueError(
          "prefetch_depth must be less than 256, but it is {}".format(
              prefetch_depth))

    if device_ordinal is None:
      strategy = ds_context.get_strategy()
      if isinstance(strategy, ipu_strategy.IPUStrategyV1):
        device_ordinal = strategy._device_ordinal  # pylint: disable=protected-access
      else:
        device_ordinal = 0

    if not isinstance(device_ordinal, int):
      raise ValueError('Device ordinal must be an integer')

    if device_ordinal < 0:
      raise ValueError('Device ordinal must be >= 0')

    with ops.device('/device:CPU:0'):
      self._dataset = dataset
      self._structure = dataset_ops.get_structure(self._dataset)
      self._flat_structure = dataset._flat_structure
      self._device_ordinal = device_ordinal
      self._prefetch_depth = prefetch_depth

      # Apply the dataset options - do this before replica handling to make sure
      # all the optimizations can be applied.
      self._dataset = self._dataset._apply_options()  # pylint: disable=protected-access

      # ID used for differentiating between datasets.
      self._id = kwargs[
          _internal_id] if _internal_id in kwargs else _generate_unique_name()

      try:
        ds_variant = self._dataset._variant_tensor  # pylint: disable=protected-access
      except TypeError:
        ds_variant = self._dataset._as_variant_tensor  # pylint: disable=protected-access

      if not context.executing_eagerly():
        # For Estimators, the graph can be frozen before the estimator calls
        # the initilizer or deleter methods.  So we need to create the
        # initialize and delete operations early.  For eager execution in
        # TF2, the operations execute eagerly, so they don't exist in any
        # frozen graph.
        with ops.colocate_with(ds_variant):
          self._init_op = gen_pop_datastream_ops.ipu_create_dataset_iterator(
              input_dataset=ds_variant,
              feed_id=self._id,
              device_ordinal=self._device_ordinal,
              **self._dataset._flat_structure)  # pylint: disable=protected-access

        self._deleter = gen_pop_datastream_ops.ipu_delete_dataset_iterator(
            feed_id=self._id, device_ordinal=self._device_ordinal)

    self._dequeued = False
    self._initialized = False
    self._from_spec = False

  def _dequeue(self):
    """Returns a nested structure of `tf.Tensor`s representing the next element
    in the infeed queue.

    This function should not be called directly, instead the infeed should be
    passed to a loop from `tensorflow.python.ipu.loops`.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    flat_ret = gen_pop_datastream_ops.pop_datastream_infeed_dequeue(
        feed_id=self._id,
        prefetch_depth=self._prefetch_depth,
        **self._flat_structure)
    self._dequeued = True
    return structure.from_tensor_list(self._structure, flat_ret)

  @property
  def dequeued(self):
    """Returns whether this queue has been dequeued.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    return self._dequeued

  @property
  def number_of_tuple_elements(self):
    """Returns the number of arguments supplied by this IPUInfeedQueue."""
    args, kwargs = loops._body_arguments(self._structure)  # pylint: disable=protected-access
    return len(args) + len(kwargs)

  @property
  def initializer(self):
    """A `tf.Operation` that should be run to initialize this IPUInfeedQueue.

    Returns:
      A `tf.Operation` that should be run to initialize this IPUInfeedQueue

    Raises:
      ValueError: if the function `initializer` has already been called.
    """
    if self._from_spec:
      raise RuntimeError(
          "IPUInfeedQueue created from spec should already be initialized.")

    if context.executing_eagerly():
      try:
        ds_variant = self._dataset._variant_tensor  # pylint: disable=protected-access
      except TypeError:
        ds_variant = self._dataset._as_variant_tensor  # pylint: disable=protected-access

      with ops.colocate_with(ds_variant):
        self._initialized = True
        return gen_pop_datastream_ops.ipu_create_dataset_iterator(
            input_dataset=ds_variant,
            feed_id=self._id,
            device_ordinal=self._device_ordinal,
            **self._dataset._flat_structure)  # pylint: disable=protected-access

    if self._initialized:
      raise ValueError(
          """The IPUInfeedQueue `initializer` function can only be accessed once."""
      )
    self._initialized = True
    return self._init_op

  @property
  def deleter(self):
    """A `tf.Operation` that can be run to delete the resources owned
    by this IPUInfeedQueue. This allows creating a new IPUInfeedQueue
    with the same name afterwards.

    Returns:
      A `tf.Operation` that can be run to delete this IPUInfeedQueue
    """
    if self._from_spec:
      raise RuntimeError("IPUInfeedQueue created from spec cannot be deleted.")

    if context.executing_eagerly():
      return gen_pop_datastream_ops.ipu_delete_dataset_iterator(
          feed_id=self._id, device_ordinal=self._device_ordinal)

    return self._deleter

  def get_next(self):
    """Obsolete function."""
    raise ValueError("""`get_next()` is now obsolete as the IPUInfeedQueue is \
now automatically dequeued by the loop.""")

  @property
  def _type_spec(self):
    if not self._initialized:
      raise RuntimeError(
          "Spec for IPUInfeedQueue can only be created for already initialized "
          "queues.")

    return _IPUInfeedQueueSpec(self._id, self._structure, self._flat_structure,
                               self._device_ordinal, self._prefetch_depth)

  @classmethod
  def _from_type_spec(cls, spec):
    """Function for creating an IPUInfeedQueue from a serialized spec. A
    serialized spec implies that the IPUInfeedQueue has been initialized and it
    also does not delete the IPUInfeedQueue.
    """

    obj = cls.__new__(cls)
    # pylint: disable=protected-access
    obj._id = spec._id
    obj._structure = spec._structure
    obj._flat_structure = spec._flat_structure
    obj._device_ordinal = spec._device_ordinal
    obj._prefetch_depth = spec._prefetch_depth
    obj._initialized = True
    obj._dequeued = True
    obj._from_spec = True
    # pylint: enable=protected-access
    return obj


# pylint: disable=abstract-method
class _IPUInfeedQueueSpec(type_spec.TypeSpec):
  """Type specification for `IPUInfeedQueue`.

  Allows for the IPUInfeedQueue to be recreated (this instance however does not
  own the underlying dataset and hence can't instantiate or delete an
  IPUInfeedQueue).
  """

  __slots__ = [
      "_id", "_structure", "_flat_structure", "_device_ordinal",
      "_prefetch_depth"
  ]

  def __init__(self, feed_id, feed_structure, feed_flat_structure,
               feed_device_ordinal, feed_prefetch_depth):
    self._id = feed_id
    self._structure = feed_structure
    self._flat_structure = feed_flat_structure
    self._device_ordinal = feed_device_ordinal
    self._prefetch_depth = feed_prefetch_depth

  @property
  def value_type(self):
    return IPUInfeedQueue

  def _serialize(self):
    return (self._id, self._structure, self._flat_structure,
            self._device_ordinal, self._prefetch_depth)

  @property
  def _component_specs(self):
    return tuple()

  def _to_components(self, value):
    del value
    return tuple()

  def _from_components(self, components):
    del components
    return IPUInfeedQueue._from_type_spec(self)  # pylint: disable=protected-access

  @staticmethod
  def from_value(value):
    return _IPUInfeedQueueSpec(value.id, value.structure, value.flat_structure,
                               value.device_ordinal, value.prefetch_depth)


class IPUIterator(iterator_ops.OwnedIterator):
  """An IPU specific iterator producing tf.Tensor objects from a
  tf.data.Dataset.

  This iterator should be initially constructed in eager mode in order to make
  sure that the dataset is constructed on a compatible device.

  Note that the infeed queue is not deleted.

  The elements from iterator can only be accessed inside of tf.functions for
  maximum performance.
  """
  def __init__(self,
               dataset=None,
               infeed_spec=None,
               element_spec=None,
               **kwargs):
    """Creates a new iterator from the given dataset.

    If `dataset` is not specified, the iterator will be created from the given
    infeed spec and element structure. In particular, the alternative for
    constructing the iterator is used when the iterator is reconstructed from
    it `CompositeTensor` representation.

    Args:
      dataset: A `tf.data.Dataset` object.
      infeed_spec: IPUInfeedQueue `TypeSpec` the iterator from.
      element_spec: A nested structure of `TypeSpec` objects that
        represents the type specification of elements of the iterator.
      **kwargs: Arguments passed to the `IPUInfeedQueue`.

    Raises:
      ValueError: If `dataset` is not provided and either `infeed_spec` or
        `element_spec` is not provided. Or `dataset` is provided and either
        `infeed_spec` and `element_spec` is provided.
    """
    # Call the grandparent class skipping OwnedIterator as the specs are
    # different.
    super(iterator_ops.OwnedIterator, self).__init__()  # pylint: disable=bad-super-call

    error_message = ("Either `dataset` or both `infeed_spec` and "
                     "`element_spec` need to be provided.")
    if dataset is None:
      if (infeed_spec is None or element_spec is None):
        raise ValueError(error_message)
      # pylint: disable=protected-access
      self._element_spec = element_spec
      self._flat_output_types = structure.get_flat_tensor_types(
          self._element_spec)
      self._flat_output_shapes = structure.get_flat_tensor_shapes(
          self._element_spec)
      self._infeed_queue = infeed_spec._from_components(None)
    else:
      if (infeed_spec is not None or element_spec is not None):
        raise ValueError(error_message)
      self._create_iterator(dataset, **kwargs)

    self._infeed_spec = self._infeed_queue._type_spec  # pylint: disable=protected-access

  def _create_iterator(self, dataset, **kwargs):  # pylint: disable=arguments-differ
    self._infeed_queue = IPUInfeedQueue(dataset, **kwargs)
    # Run the initializer.
    self._infeed_queue.initializer  # pylint: disable=pointless-statement

    self._element_spec = dataset.element_spec
    self._flat_output_types = structure.get_flat_tensor_types(
        self._element_spec)
    self._flat_output_shapes = structure.get_flat_tensor_shapes(
        self._element_spec)

  def __iter__(self):
    return self

  def next(self):
    return self.__next__()

  def _next_internal(self):
    if context.executing_eagerly():
      raise RuntimeError(
          "Accessing dataset elements can not be accessed in eager mode when "
          "inside 'IPUStrategy'. The dataset elements should be accessed "
          "inside of a 'tf.function' with 'experimental_compile=True' set in "
          "order to achieve optimal performance. See documentation for "
          "detailed examples. If this behavior is required please set "
          "'enable_dataset_iterators=False' when creating an 'IPUStrategy'.")
    return self._infeed_queue._dequeue()  # pylint: disable=protected-access

  @property
  def _type_spec(self):
    return _IPUIteratorSpec(self.infeed_spec, self.element_spec)

  def __next__(self):
    try:
      return self._next_internal()
    except errors.OutOfRangeError:
      raise StopIteration

  @property
  def element_spec(self):
    return self._element_spec

  @property
  def infeed_spec(self):
    return self._infeed_spec

  def get_next(self):
    return self._next_internal()

  def get_next_as_optional(self):
    raise NotImplementedError(
        'IPU Dataset iterator does not support get_next_as_optional().')

  def _gather_saveables_for_checkpoint(self):
    raise NotImplementedError(
        'IPU Dataset iterator is not currently saveable.')


class IPUOwnedIterator(IPUIterator):
  """An IPU specific iterator producing tf.Tensor objects from a
  tf.data.Dataset.

  The iterator resource created through `IPUOwnedIterator` is owned by the
  Python object and the life time of the underlying resource is tied to the life
  time of the `IPUOwnedIterator` object. This makes `IPUOwnedIterator`
  appropriate for use inside of tf.functions.

  This iterator should be initially constructed in eager mode in order to make
  sure that the dataset is constructed on a compatible device.

  The elements from iterator can only be accessed inside of tf.functions for
  maximum performance.
  """
  def __init__(self,
               dataset=None,
               infeed_spec=None,
               element_spec=None,
               **kwargs):
    """Creates a new iterator from the given dataset.

    If `dataset` is not specified, the iterator will be created from the given
    infeed spec and element structure. In particular, the alternative for
    constructing the iterator is used when the iterator is reconstructed from
    it `CompositeTensor` representation.

    Args:
      dataset: A `tf.data.Dataset` object.
      infeed_spec: IPUInfeedQueue `TypeSpec` the iterator from.
      element_spec: A nested structure of `TypeSpec` objects that
        represents the type specification of elements of the iterator.
      **kwargs: Arguments passed to the `IPUInfeedQueue`.

    Raises:
      ValueError: If `dataset` is not provided and either `infeed_spec` or
        `element_spec` is not provided. Or `dataset` is provided and either
        `infeed_spec` and `element_spec` is provided.
    """
    super().__init__(dataset=dataset,
                     infeed_spec=infeed_spec,
                     element_spec=element_spec,
                     **kwargs)

  def _create_iterator(self, dataset, **kwargs):
    super()._create_iterator(dataset, **kwargs)

    # Create a deleter which gets called when the dataset/infeed owning instance
    # of the iterator goes out of scope.
    self._deleter = _IPUOwnedIteratorDeleter(
        self._infeed_queue._id, self._infeed_queue._device_ordinal)  # pylint: disable=protected-access

  @property
  def _type_spec(self):
    return _IPUOwnedIteratorSpec(self.infeed_spec, self.element_spec)


# pylint: disable=abstract-method
class _IPUIteratorSpec(type_spec.TypeSpec):
  """Type specification for `IPUIterator`.

  For instance, `_IPUIteratorSpec` can be used to define a tf.function that
  takes `IPUIterator` as an input argument:

  >>> @tf.function(experimental_compile=True)
  ... def square(iterator):
  ...   x = iterator.get_next()
  ...   return x * x
  >>> dataset = tf.data.Dataset.from_tensors(5)
  >>> iterator = iter(dataset)
  >>> print(square(iterator))
  tf.Tensor(25, shape=(), dtype=int32)
  """

  __slots__ = ["_infeed_spec", "_element_spec"]

  def __init__(self, infeed_spec, element_spec):
    self._infeed_spec = infeed_spec
    self._element_spec = element_spec

  @property
  def value_type(self):
    return IPUIterator

  def _serialize(self):
    return (self._infeed_spec, self._element_spec)

  @property
  def _component_specs(self):
    return tuple()

  def _to_components(self, value):
    del value
    return tuple()

  def _from_components(self, components):
    del components
    return IPUIterator(infeed_spec=self._infeed_spec,
                       element_spec=self._element_spec)

  @staticmethod
  def from_value(value):
    return _IPUIteratorSpec(  # pylint: disable=protected-access
        value._infeed_spec, value.element_spec)  # pylint: disable=protected-access


# pylint: disable=abstract-method
class _IPUOwnedIteratorSpec(_IPUIteratorSpec):
  """Type specification for `IPUOwnedIterator`.

  For instance, `_IPUOwnedIteratorSpec` can be used to define a tf.function that
  takes `IPUOwnedIterator` as an input argument:

  >>> @tf.function(experimental_compile=True)
  ... def square(iterator):
  ...   x = iterator.get_next()
  ...   return x * x
  >>> dataset = tf.data.Dataset.from_tensors(5)
  >>> iterator = iter(dataset)
  >>> print(square(iterator))
  tf.Tensor(25, shape=(), dtype=int32)
  """
  @property
  def value_type(self):
    return IPUOwnedIterator

  def _from_components(self, components):
    del components
    return IPUOwnedIterator(infeed_spec=self._infeed_spec,
                            element_spec=self._element_spec)

  @staticmethod
  def from_value(value):
    return _IPUOwnedIteratorSpec(  # pylint: disable=protected-access
        value._infeed_spec, value.element_spec)  # pylint: disable=protected-access


class _IPUOwnedIteratorDeleter(object):
  """An object which cleans up an iterator.

  An alternative to defining a __del__ method on an object. Even if the parent
  object is part of a reference cycle, the cycle will be collectable.
  """

  __slots__ = ["_handle", "_device_ordinal", "_eager_mode"]

  def __init__(self, handle, device_ordinal):
    self._handle = handle
    self._device_ordinal = device_ordinal
    self._eager_mode = context.executing_eagerly()

  def __del__(self):
    # Make sure the resource is deleted in the same mode as it was created in.
    if self._eager_mode:
      with context.eager_mode():
        gen_pop_datastream_ops.ipu_delete_dataset_iterator(
            feed_id=self._handle,
            device_ordinal=self._device_ordinal,
            asynchronous=True)
    else:
      with context.graph_mode():
        gen_pop_datastream_ops.ipu_delete_dataset_iterator(
            feed_id=self._handle, device_ordinal=self._device_ordinal)

Keras with IPUs
---------------

The Graphcore implementation of TensorFlow includes Keras support for IPUs.
Keras model creation is no different than what you would use if you were
training on other devices. To target the Poplar XLA device, Keras model creation
must be inside the ``strategy.scope`` of an ``IPUStrategy``.

For a more practical walkthrough, see `this tutorial about using Keras on the IPU <https://github.com/graphcore/tutorials/tree/sdk-release-2.5/tutorials/tensorflow2/keras>`_
from the Graphcore tutorials repository.

Single IPU models
~~~~~~~~~~~~~~~~~

You can train, evaluate or run inference on single-IPU models through the Keras
APIs as you would with other accelerators, as long as you create the model
inside the scope of an ``IPUStrategy``:

.. literalinclude:: keras_tf2_example1.py
  :language: python
  :linenos:
  :emphasize-lines: 2, 4-7, 38-39

Using steps_per_execution
~~~~~~~~~~~~~~~~~~~~~~~~~

To reduce Python overhead and maximize the performance of your model, pass in
the ``steps_per_execution`` argument to the compile method. This argument sets
the number of batches to process sequentially in a single execution. You should
increase this number to improve accelerator utilization.

.. note::

  In order to achieve best performance, ``steps_per_execution`` needs to be set
  before using ``fit()``, ``evaluate()`` and ``predict()``, even if no training
  is performed.

See the documentation for the
`compile method <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`__
for full details.

The example below highlights the usage of ``steps_per_execution``:

.. literalinclude:: keras_tf2_example2.py
  :language: python
  :linenos:
  :emphasize-lines: 48-49

Gradient accumulation
~~~~~~~~~~~~~~~~~~~~~

When training, gradient accumulation allows us to simulate bigger batch sizes.
This is achieved by accumulating the gradients across multiple batches together
then performing the weight update.

For example, if we have a model where each step is of batch size 16 and we set
`gradient_accumulation_steps_per_replica` to 4 then this simulates an input
batch of size 64.

Gradient accumulation can be easily enabled for Keras models created inside of
an ``IPUStrategy`` by calling the
:py:meth:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension.set_gradient_accumulation_options`
method for Functional Keras models and the
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_gradient_accumulation_options`
method for Sequential Keras models. See the respective method documentation
for more details.

.. note::

  When using data-parallelism, the ``steps_per_execution`` value the model was
  compiled with must be an integer multiple of
  ``gradient_accumulation_steps_per_replica`` multiplied by the number of
  replicas in the model. Data parallelism is discussed in
  :numref:`automatic-data-parallelism`.


.. note::

  Not all operations are compatible with gradient accumulation.

The example below highlights the usage of ``set_gradient_accumulation_options``:

.. literalinclude:: keras_tf2_example3.py
  :language: python
  :linenos:
  :emphasize-lines: 51-52

Model parallelism
~~~~~~~~~~~~~~~~~

The models described so far occupy a single IPU device, however some models
might require the model layers to be split across multiple IPU devices to
achieve high compute efficiency.

One method to achieve model parallelism is called *pipelining*, where the
model layers are assigned to *pipeline stages*. Each pipeline stage can be
assigned to a different device and different devices can execute in parallel.

The method to pipeline your model depends on whether your model is a
``Sequential`` model, a ``Functional`` model, or is subclassed from the ``Model``
class.

Sequential model
________________

To enable IPU pipelining for a ``Sequential`` model (an instance of
`tensorflow.keras.Sequential`), a list of per-layer pipeline stage
assignments should be passed to the
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_pipeline_stage_assignment`
method of the model.

For example, a simple four layer ``Sequential`` model could be assigned to two
different pipeline stages as follows:

.. literalinclude:: keras_tf2_example7.py
  :language: python
  :linenos:
  :start-at: model = tf.keras.Sequential([
  :end-at: model.set_pipeline_stage_assignment([0, 0, 1, 1])

You can confirm which layers are assigned to which stages using the
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.print_pipeline_stage_assignment_summary`
method of the model.

Functional model
________________

There are two ways to enable IPU pipelining for a ``Functional`` model (an
instance of `tensorflow.keras.Model`) depending on if you're pipelining a model
you are writing yourself or an existing model.

Pipelining a model you are writing yourself
===========================================

To pipeline a ``Functional`` model you are writing yourself, each layer call
must happen within the scope of an `ipu.keras.PipelineStage` context.

For example, a simple four layer ``Functional`` model could be assigned to two
different pipeline stages as follows:

.. literalinclude:: keras_tf2_example8.py
  :language: python
  :linenos:
  :start-at: input_layer = tf.keras.layers.Input((28, 28))
  :end-at: model = tf.keras.Model(inputs=input_layer, outputs=x)

.. note::
Layers *constructed* within an `ipu.keras.PipelineStage` context will have that
pipeline stage assigned to all invocations of the layer. These assignments are
overridden if the layer calls happen within a different
`ipu.keras.PipelineStage` context.

Pipelining an existing functional model
=======================================

To pipeline an existing ``Functional`` model, you can use
:py:meth:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension.get_pipeline_stage_assignment`.
Each layer invocation in the model has an associated
:py:class:`~tensorflow.python.ipu.keras.extensions.FunctionalLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.
`get_pipeline_stage_assignment` returns a list of these stage assignments,
which you can inspect and modify. Note that the list is in post-order, which
means the assignments are returned in the order they will be executed.

Once you are done modifying the stage assignments, you should use
:py:meth:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension.set_pipeline_stage_assignment`
to set them on the model.

For example, a naive way of pipelining ResNet50 would be to assign everything
up until the "conv4_block2_add" layer invocation to the first stage, then
everything else to the second stage, as follows:

.. literalinclude:: keras_tf2_example9.py
  :language: python
  :linenos:
  :start-at: strategy = ipu.ipu_strategy.IPUStrategy()

.. note::

  You can use :py:meth:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension.print_pipeline_stage_assignment_summary`
  to print the pipeline stage assignments of the model's layer invocations.

.. note::

  This method of assigning pipeline stages can also be used with ``Functional``
  models you are writing yourself, as well as with ``Sequential``
  models and ``Model`` subclasses using the
  :py:class:`~tensorflow.python.ipu.keras.extensions.SequentialExtension` and
  :py:class:`~tensorflow.python.ipu.keras.extensions.ModelExtension`
  equivalents.

Model subclass
______________

``Model`` subclasses are subclasses of `tf.keras.Model`, which override the call
method. There are two ways to enable IPU pipelining for an instance of a
``Model`` subclass, depending on if you're pipelining a model you are writing
yourself or an existing model. These are very similar to the methods available
for ``Functional`` models.

Pipelining a model you are writing yourself
===========================================

To pipeline a ``Model`` subclass you are writing yourself, each layer call
must happen within the scope of an `ipu.keras.PipelineStage` context.

For example, a simple four layer ``Model`` subclass could be assigned to four
different pipeline stages as follows:

.. literalinclude:: keras_tf2_example11.py
  :language: python
  :linenos:
  :start-at: class MyModel(tf.keras.Model):
  :end-at:     return x

.. note::
Layers *constructed* within an `ipu.keras.PipelineStage` context will have that
pipeline stage assigned to all invocations of the layer. These assignments are
overridden if the layer calls happen within a different
`ipu.keras.PipelineStage` context.

Pipelining an existing model
============================

To pipeline an existing ``Model`` subclass, you must use
:py:meth:`~tensorflow.python.ipu.keras.extensions.ModelExtension.get_pipeline_stage_assignment`.
Each layer invocation in the model has an associated
:py:class:`~tensorflow.python.ipu.keras.extensions.ModelLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.
`get_pipeline_stage_assignment` returns a list of these stage assignments,
which you can inspect and modify. Note that the list is in post-order, which
means the assignments are returned in the order they will be executed.

Once you are done modifying the stage assignments, you should use
:py:meth:`~tensorflow.python.ipu.keras.extensions.ModelExtension.set_pipeline_stage_assignment`
to set them on the model.

Before you can get or set pipeline stage assignments, you must first call
:py:meth:`keras.Model.build` on your model, specifying the input shapes.
This traces the model's call function using the shapes specified. The resulting
graph is what will be used for pipelined execution. You can update the graph by
calling build again, though this will invalidate existing pipeline stage
assignments if the structure of the updated graph is different.

.. note::

  If you need to specify input dtypes when calling :py:meth:`keras.Model.build`,
  you can pass in :py:class:`keras.Input` objects instead of plain shapes.

For example, an existing ``Model`` subclass with four layers, could be assigned
to four different pipeline stages as follows:

.. literalinclude:: keras_tf2_example12.py
  :language: python
  :linenos:
  :start-at:   model = ExistingModel()
  :end-at:   model.set_pipeline_stage_assignment(assignments)

.. note::

  You can use :py:meth:`~tensorflow.python.ipu.keras.extensions.ModelExtension.print_pipeline_stage_assignment_summary`
  to print the pipeline stage assignments of the model's layer invocations.

.. note::

  This method of assigning pipeline stages can also be used with ``Model``
  subclasses you are writing yourself, as well as with ``Functional`` and
  ``Sequential`` models using the
  :py:class:`~tensorflow.python.ipu.keras.extensions.SequentialExtension` and
  :py:class:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension`
  equivalents.


.. _automatic-data-parallelism:

Automatic data parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

IPU TensorFlow supports automatic data parallelism when multiple IPU devices are
configured with the system. Automatic data parallelism is achieved by model
replication across available IPU devices. The number of times the model is
replicated is called the replication factor; higher replication factors allow
higher data throughput.

When replicating, gradients are reduced across replicas during training, which
has implications for gradient accumulation. For a non replicated model, the
*effective batch size* is the product of the dataset batch size and the number
of gradient accumulation steps. In the case of a replication factor greater than
one, the *effective batch size* is additionally scaled by the replication
factor according to the following formula:

`effective_batch_size = dataset_batch_size * gradient_accumulation_steps_per_replica * num_replicas`

Asynchronous callbacks
~~~~~~~~~~~~~~~~~~~~~~

IPU TensorFlow supports the use of ``Callback`` objects with the Keras APIs,
however there is an important difference to note when specifying
`steps_per_execution`. In IPU TensorFlow, if `steps_per_execution` is specified
for your model, then per-batch callback functions will only be invoked every
`steps_per_execution` steps, which can have the effect of delaying access to
results.

However, IPU TensorFlow also supports *asynchronous callbacks* by providing a
polling mechanism which allows results to be accessed at the earliest possible
instance. Asynchronous callbacks can be enabled by invoking
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_asynchronous_callbacks`
with `True` on your ``Sequential`` or ``Functional`` Keras model.

Configuring Infeeds and Outfeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keras models created inside of an ``IPUStrategy`` scope automatically create
``IPUInfeedQueue`` and ``IPUOutfeedQueue`` data queues for efficiently feeding
data to and from the IPU devices when using ``fit()``, ``evaluate()`` and
``predict()``.

Instances of ``IPUInfeedQueue`` and ``IPUOutfeedQueue`` can be created with
optional arguments which can affect performance of the model.

For configuring the ``IPUInfeedQueue`` use
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_infeed_queue_options`
on your ``Sequential`` or ``Functional`` Keras model.

For configuring the ``IPUOutfeedQueue`` use
:py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_outfeed_queue_options`
on your ``Sequential`` or ``Functional`` Keras model.

For example the ``prefetch_depth`` parameter of the ``IPUInfeedQueue`` and the
``buffer_depth`` parameter of the ``IPUOutfeedQueue`` can be configured as
follows:

.. literalinclude:: keras_tf2_example10.py
  :language: python
  :linenos:
  :emphasize-lines: 26-28


Saving and loading Keras models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saving and loading a Keras model must be done within the IPUStrategy scope in
order to save/load IPU-specific information.

.. note::
  The arguments `pipelining_kwargs` from :py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_pipelining_options` and
  `gradient_accumulation_optimizer_kwargs` from :py:meth:`~tensorflow.python.ipu.keras.extensions.SequentialExtension.set_gradient_accumulation_options`
  are not serializable, which means that when the model
  is being saved, their values are not saved. When restoring/loading a model,
  call ``set_pipelining_options()`` or ``set_gradient_accumulation_options()``
  again.


Implementation details
~~~~~~~~~~~~~~~~~~~~~~

When instantiating a standard TensorFlow Keras model inside the scope of
an `IPUStrategy` instance, it is dynamically injected with additional,
IPU-specific, functions.
This is done through the relevant *IPU Keras extension classes*.
For `tensorflow.keras.Sequential`, IPU-specific extensions exist in
:py:class:`~tensorflow.python.ipu.keras.extensions.SequentialExtension` and for
`tensorflow.keras.Model` in
:py:class:`~tensorflow.python.ipu.keras.extensions.FunctionalExtension`.

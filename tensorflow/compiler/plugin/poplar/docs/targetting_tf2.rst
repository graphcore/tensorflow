Support for TensorFlow 2
------------------------

In TensorFlow version 2, eager mode is enabled by default and Keras is
the main API for constructing models. Distribution strategies are the
new way of targeting different pieces of hardware.

The Graphcore implementation of TensorFlow includes IPU-specific implementations
of the ``Model`` and ``Sequential`` classes, and adds ``PipelineModel`` and
``PipelineSequential`` classes for running a model on multiple IPUs. It
also makes efficient use of the IPU by fusing operations into a single kernel
that is executed repeatedly, amortising the cost of control and I/O.

Function annotation with @tf.function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function annotation ``@tf.function`` is well documented in the standard
TensorFlow documentation. It converts the body of the annotated function into
a fused set of operations that are executed as a group, in the same way as a
whole graph would have been in TensorFlow version 1. In addition, a library
called ``autograph`` will convert Python flow control constructs into TensorFlow
graph operations.

Best practice is to ensure that anything which is intended to be executed on
the IPU is placed into a function and annotated with ``@tf.function``. This
does not apply to constructing a Keras model or using the Keras ``Model.fit()``
API. See below for details on Keras.

When calling a function that is marked with a ``@tf.function`` from within a
distribution strategy like ``IPUStrategyV1``, you should not call it directly,
but instead use the ``run`` method.

See the following online resources for more information:

- https://www.tensorflow.org/tutorials/customization/performance
- https://www.tensorflow.org/guide/function

IPUStrategyV1
~~~~~~~~~~~~~

The ``tf.distribute.Strategy`` is an API to distribute training across multiple
devices. :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategyV1` is a
subclass which targets a system with one or more IPUs attached. Another subclass,
:py:class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategyV1`,
targets a multiple system configuration.

Use the ``strategy.scope()`` context to ensure that everything within that
context will be compiled for the IPU device. You should do this instead
of using the ``tf.device`` context.

.. code-block:: python

    from tensorflow.python import ipu

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategyV1()

    with strategy.scope():
        ...

It is important to construct a Keras model within the scope of the
``IPUStrategyV1``, because Keras may create some parts of the model at
construction time, and some other parts at execution time.

See the TensorFlow documentation for more details:
https://www.tensorflow.org/guide/distributed_training

Keras
~~~~~

The Graphcore implementation of TensorFlow includes a port of Keras for the IPU,
available as :py:class:`tensorflow.python.ipu.keras`.

The Keras API is used for constructing models using a set of high-level ``Layer``
objects. See https://www.tensorflow.org/guide/keras for more information.

IPU optimized replacements for the Keras ``Model`` and ``Sequential`` classes are
available for the IPU. These have the following features:

* On-device training loop for reduction of communication overhead.
* Gradient accumulation for simulating larger batch sizes.
* Automatic data-parallelisation of the model when placed on a multi-IPU device.
  This means that during training the gradients will be reduced across
  replicas.

These are described in more detail below.

.. note::
  The model must be both instantiated and called from within an ``IPUStrategyV1``
  context.

See https://www.tensorflow.org/guide/keras/train_and_evaluate for
more background.


Model class
___________

An IPU port of the standard Keras ``Model`` is
available as :py:class:`tensorflow.python.ipu.keras.Model`.

This is a substitute for the standard Keras ``Model`` class, using only a single
IPU for training. Unlike the standard Keras ``Model`` class, it cannot
be called directly. You must use use the ``fit()``, ``evaluate()`` and
``predict()`` methods for training, evaluation and making predictions.

For a high-performance, multi-IPU solution use the
:ref:`pipeline-model`.

Sequential class
________________

An implementation of the Keras ``Sequential`` class is
available as :py:class:`tensorflow.python.ipu.keras.Sequential`.

This is a substitute for the standard Keras ``Sequential`` class, using only a
single IPU for training. For a high-performance, multi-IPU solution use
the :ref:`pipeline-sequential`.

Unlike the standard Keras ``Model`` class, it cannot be
called directly. You must use use the ``fit()``, ``evaluate()`` and
``predict()`` methods for training, evaluation and making predictions.
Similarly, you cannot get the list of trainable variables before you have
executed the model.

.. _pipeline-model:

PipelineModel class
___________________

:py:class:`~tensorflow.python.ipu.keras.PipelineModel` is an alternative for the
Keras ``Model`` class, with support for multi-device IPU pipelines. Using
pipelined execution allows the IPU to achieve high compute efficiency while
utilising multiple devices.

The ``PipelineModel`` has the same API as the standard Keras ``Model`` classes,
but will train the model on multiple IPUs and stream the data into the devices
using an ``Infeed`` queue which is created automatically.

When defining a model for use with ``PipelineModel``, the pipeline stage at
which a ``Layer`` is to be executed is given by the
:py:class:`~tensorflow.python.ipu.keras.PipelineModel` context in which it is
called.

In a machine learning model, a "step" is often considered to be one pass through
the model, in which the forward pass is done, the gradients are calculated
and then the parameters are updated. Since a pipeline accumulates multiple
gradients before applying them collectively to the parameters, we call each
of those pipeline operations a "step". So the number of data samples processed per
step is equal to the batch size multiplied by the pipeline depth.

This will be reflected in the rate at which the progress bar advances, and the
entries in the Keras history.

Like the ``Sequential`` class, ``PipelineModel`` also supports automatic
data-parallelism.


.. _pipeline-sequential:

PipelineSequential class
_____________________________

:py:class:`~tensorflow.python.ipu.keras.PipelineSequential` is an
alternative to the ``PipelineModel`` class for the Keras ``Sequential`` class.

Like the constructor for the standard Keras ``Sequential`` model,
``PipelineSequential`` takes a list of lists of layers, where each list of
layers is assigned to an IPU pipeline stage. See :ref:`tensorflow2examples` to
see how the API is used.

Like the ``Sequential`` class, ``PipelineSequential`` also supports
automatic data-parallelism.

Custom training loops
_____________________

If a more sophisticated training loop is required, then it can be described
inside a function which is marked as a ``@tf.function``. See :ref:`tensorflow2examples`
for an example.

The outer training function should be called using the ``run`` method on the
``IPUStrategyV1`` object, to ensure that it is executed using the strategy's
configuration.

.. note::
  It is not possible to use either ``PipelineModel`` or
  ``PipelineSequential`` in a custom training loop.

For more information on the ``@tf.function`` annotation, see the
`TensorFlow function documentation <https://www.tensorflow.org/guide/function>`_.

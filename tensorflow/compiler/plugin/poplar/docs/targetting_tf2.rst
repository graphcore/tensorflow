Targeting the IPU with TensorFlow 2
-----------------------------------

In TensorFlow version 2, the Eager mode is enabled by default, and Keras has
become the main API for constructing models. Distribution strategies are the
new way of targeting different pieces of hardware.

As in TensorFlow version 1, there are a small number of things
that need to be done when constructing and executing a model in order to
target the IPU efficiently. The IPU achieves its performance by fusing
operations into a single kernel that is executed repeatedly, amortising
the cost of control and I/O.

Function annotation with @tf.function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function annotation ``@tf.function`` is well documented in the standard
TensorFlow documentation. It converts the body of the annotated function into
a fused set of operations that are executed as a group, in the same way as a
whole graph would have been in TensorFlow version 1. In addition, a library
called ``autograph`` will convert python flow control constructs into TensorFlow
graph operations.

Best practice is to ensure that anything which is intended to be executed on
the IPU is placed into a function and annotated with ``@tf.function``. This
does not apply to constructing a Keras model or using the Keras ``Model.fit()``
API. See below for details on Keras.

When calling a function that is marked with a ``@tf.function`` from within a
distribution strategy like ``IPUStrategy``, you should not call them directly,
but instead use the ``experimental_run_v2`` method.

See the following online resources for more information.

- https://www.tensorflow.org/tutorials/customization/performance
- https://www.tensorflow.org/guide/function

IPUStrategy
~~~~~~~~~~~

Distribution strategies are a more advanced and flexible version of device
tagging. The ``IPUStrategy`` is a sub-class of distribution strategy which
specifically targets a system with one or more IPUs attached. A separate
class ``IPUMultiWorkerStrategy`` is for targeting a multiple system
configuration.

Use the ``strategy.scope()`` context to ensure that everything within that
context will be compiled for the IPU device. You should do this instead
of using the ``tf.device`` context.

.. code-block:: python

    from tensorflow.python import ipu

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        ...

It is important to construct any Keras model within the scope of the
``IPUStrategy``, because a Keras ``Model`` class may create some of the model at
construction time, and some other parts of it at execution time.

See the online documentation for more details.

- https://www.tensorflow.org/guide/distributed_training

Keras
~~~~~

The Keras API is used for constructing models using a set of high-level ``Layers``
objects. https://www.tensorflow.org/guide/keras.

Full support is available for Keras on the IPU. It is important to ensure
that the model is both instantiated and called from within an ``IPUStrategy``
context.

- https://www.tensorflow.org/guide/keras/train_and_evaluate

IPU optimized drop-in replacements for Keras Model and Keras Sequential are
available and described below.

Model class
___________

A higher performance alternative to using the standard Keras Model is
available. It is called ``Model``, and found at
``tensorflow.python.ipu.keras.Model``. It supports the following features:

* On device training loop for reduction of communication overhead.
* Gradient accumulation for simulating larger batch sizes.
* Automatic data-parallelism of the model when placed on a multi-IPU device,
  which means that during training the gradients will be reduced across
  replicas.

It is a substitute for the Keras Model class, when only a single IPU
is used for training. For a high performance multi-IPU solution use the
``PipelineModel`` described below.

Unlike the standard Keras model classes, it must be trained, evaluated and
operated with the ``fit``, ``evaluate`` and ``predict`` methods. It cannot be
called directly.

Sequential class
________________

A higher performance alternative to using the standard Keras Sequential is
available. It is called ``Sequential``, and found at
``tensorflow.python.ipu.keras.Sequential``. It supports the following features:

* On device training loop for reduction of communication overhead.
* Gradient accumulation for simulating larger batch sizes.
* Automatic data-parallelism of the model when placed on a multi-IPU device,
  which means that during training the gradients will be reduced across
  replicas.

It is a substitute for the Keras Sequential class, when only a single IPU
is used for training. For a high performance multi-IPU solution use the
``SequentialPipelineModel`` described below.

Unlike the standard Keras model classes, it must be trained, evaluated and
operated with the ``fit``, ``evaluate`` and ``predict`` methods. It cannot be
called directly. For a similar reason, you cannot get the list of trainable
variables before you have executed it.

PipelineModel and SequentialPipelineModel classes
_________________________________________________

``PipelineModel`` and ``SequentialPipelineModel`` are substitutes for the Keras
Model and Sequential model classes (respectively), with support for multi-device
IPU pipelines. Using pipelined execution allows the IPU to achieve high compute
efficiency while utilising multiple devices.

PipelineModel and SequentialPipelineModel have the same APIs as the standard Keras
Model and Sequential classes, but will train the model on multiple IPUs and stream
the data into the devices using an Infeed queue which is created automatically.

When defining a graph for use with PipelineModel, the stage at which a node (or
set of nodes) is to be executed is given by the ``PipelineStage`` context in
which it is created.

The constructor of SequentialPipelineModel takes, rather than a list of layers as
with the standard Sequential model, a list of lists of layers, one for each IPU
pipeline stage. 

See the examples section to see how the APIs of each are used.

In a machine learning model a step is often considered to be one pass through
the model where the forward pass is done, then the gradients are calculated
and then the parameters are updated. Since a pipeline accumulates multiple
gradients before applying them collectively to the parameter, we call a step
one of those pipeline operations. So the number of data samples processed per
step is equal to the batch size multiplied by the pipeline depth.

This will be reflected in the rate at which the progress bar advances, and the
entries in the Keras History.

Note that ``PipelineModel`` and ``SequentialPipelineModel`` also support
automatic data parallelism, as with their non-pipelined counterparts.

Custom training loops
_____________________

If a more sophisticated training loop is required, then it can be described
inside a function which is marked as a ``@tf.function``. See the examples
section for a full example.

The outer training function should be called using the ``experimental_run_v2``
method on the ``IPUStrategy`` object, to ensure that it is executed using the
strategy's configuration.

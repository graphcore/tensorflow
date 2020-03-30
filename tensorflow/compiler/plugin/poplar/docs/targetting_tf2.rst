Targeting the IPU with TensorFlow 2
-----------------------------------

In TensorFlow version 2, the Eager mode is enabled by default, and Keras has
become the main API for constructing models.  Distribution strategies are the
new way of targeting different pieces of hardware.

As in TensorFlow version 1, there are a small number of things
that need to be done when constructing and executing a model in order to
target the IPU efficiently. The IPU achieves its performance by fusing
operations into a single kernel that is executed repeatedly, amortising
the cost of control and I/O.

IPUStrategy
~~~~~~~~~~~

Distribution strategies are a more advanced and flexible version of device
tagging. The ``IPUStrategy`` is a sub-class of distribution strategy which
specifically targets a system with one or more IPUs attached.  A separate
class ``IPUMultiWorkerStrategy`` is for targeting a multiple system
configuration.

Use the ``strategy.scope()`` context to ensure that everything within that
context will be compiled for the IPU device.  You should do this instead
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

Function annotation with @tf.function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function annotation ``@tf.function`` is well documented in the standard
TensorFlow documentation.  It converts the body of the annotated function into
a fused set of operations that are executed as a group, in the same way as a
whole graph would have been in TensorFlow version 1.  In addition, a library
called ``autograph`` will convert python flow control constructs into TensorFlow
graph operations.

Best practice is to ensure that anything which is intended to be executed on
the IPU is placed into a function and annotated with ``@tf.function``.  This
does not apply to constructing a Keras model or using the Keras ``Model.fit()``
API.  See below for details on Keras.

When calling a function that is marked with a ``@tf.function`` from within a
distribution strategy like ``IPUStrategy``, you should not call them directly,
but instead use the ``experimental_run_v2`` method.

See the following online resources for more information.

- https://www.tensorflow.org/tutorials/customization/performance
- https://www.tensorflow.org/guide/function

Keras
~~~~~

The Keras API is used for constructing models using a set of high-level ``Layers``
objects.  https://www.tensorflow.org/guide/keras.

Full support is available for Keras on the IPU.  It is important to ensure
that the model is both instantiated and called from within an ``IPUStrategy``
context.

- https://www.tensorflow.org/guide/keras/train_and_evaluate

The Model.fit method
____________________

This method of the Keras ``Model`` class can be used within an ``IPUStrategy``
to train a model without the need for a specialised training loop.

For high performance training, the ``fit`` API should be avoided, because it
does not provide an on-device training loop.

Custom training loops
_____________________

If a more sophisticated training loop is required, then it can be described
inside a function which is marked as a ``@tf.function``.  See the examples
section for a full example.

The outer training function should be called using the ``experimental_run_v2``
method on the ``IPUStrategy`` object, to ensure that it is executed using the
strategy's configuration.

PipelinedModel
______________

``PipelinedModel`` is a substitute for the Keras Sequential model class, with
support for multi-device IPU pipelines.  Using pipelined execution allows the
IPU to achieve high compute efficiency while utilising multiple devices.

The PipelinedModel has the same API as the standard Keras Model and
Sequential classes, but will train the model on multiple IPUs and stream
the data into the devices using an Infeed queue which is created automatically.

The constructor takes, rather than a list of layers as with the standard
Sequential model, a list of lists of layers, one for each IPU pipeline stage.
See the examples section to see how the API is used.

In a machine learning model a step is often considered to be one pass through
the model where the forward pass is done, then the gradients are calculated
and then the parameters are updated.  Since a pipeline accumulates multiple
gradients before applying them collectively to the parameter, we call a step
one of those pipeline operations.  So the number of data samples processed per
step is equal to the batch size multiplied by the pipeline depth.

This will be reflected in the rate at which the progress bar advances, and the
entries in the Keras History.
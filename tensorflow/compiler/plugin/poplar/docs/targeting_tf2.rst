Support for TensorFlow 2
------------------------

In TensorFlow version 2, eager mode is enabled by default and Keras is
the main API for constructing models. Distribution strategies are the
new way of targeting different pieces of hardware.

IPUStrategy
~~~~~~~~~~~

The ``tf.distribute.Strategy`` is an API to distribute training across multiple
devices. :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy` is a
subclass which targets a single system with one or more IPUs attached. Another
subclass, :py:class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategyV1`,
targets a distributed system with multiple machines (workers). For more
information, see the :any:`distributed training <distributed_training>` section.

Use the ``strategy.scope()`` context to ensure that everything within that
context will be compiled for the IPU device. You should do this instead
of using the ``tf.device`` context:

.. code-block:: python

    from tensorflow.python import ipu

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        ...

.. note::

    It is important to *construct* a Keras model within the scope of the
    ``IPUStrategy``, because Keras may create some parts of the model at
    construction time, and some other parts at execution time.

See the TensorFlow documentation for more details on distribution strategies:
https://www.tensorflow.org/guide/distributed_training

Execution modes
~~~~~~~~~~~~~~~

TensorFlow operations can be executed in either **graph mode** or
**eager mode**. Both of these modes are supported on IPUs, however graph mode is
much more efficient. It is therefore important to understand the difference
between them and understand how to write TensorFlow programs which will fully
utilize the IPU devices.

Graph mode with @tf.function
____________________________

The TensorFlow function annotation ``@tf.function`` converts the body of the
annotated function into a fused set of operations that are executed as a group,
in the same way as a whole graph would have been in TensorFlow version 1. In
addition, a library called ``autograph`` will convert Python control flow
constructs into TensorFlow graph operations.

It is best practice to ensure that anything which is intended to be executed on
the IPU is placed into a Python function which is annotated with
``@tf.function(jit_compile=True)``. Note that this does not apply to
constructing a Keras model or using the Keras ``Model.fit()`` API. See the
:any:`Keras with IPUs <keras_tf2>` section for details on Keras.

When calling a function which is marked with a
``@tf.function(jit_compile=True)`` annotation from within a
distribution strategy such as ``IPUStrategy``, you should not call it directly,
but instead use the ``run`` method. For example:

.. literalinclude:: targeting_tf2_example1.py
  :language: python
  :linenos:
  :emphasize-lines: 21

.. note::

  When using the ``@tf.function`` annotation, it is important to set the
  ``jit_compile=True`` argument to ensure best performance.

For more information about ``tf.function`` and examples, see the TensorFlow
documentation at https://www.tensorflow.org/guide/function.

Eager mode
__________

Eager mode is the default execution mode for TensorFlow operations. This mode is
supported on IPUs, however it is not as performant as graph mode and **we do not
recommend using it**.

For example, the code below executes the ``tf.matmul`` immediately on an IPU
device and returns a ``tf.Tensor`` object containing the result:

.. literalinclude:: targeting_tf2_example2.py
  :language: python
  :linenos:

On-device loops
~~~~~~~~~~~~~~~

In the :any:`Keras with IPUs <keras_tf2>` section, we describe how to use Keras
to perform training, testing and prediction. However, sometimes a more
sophisticated loop is required. You can create these to train, test and run
inference of your models using a loop created inside of a ``tf.function`` - this
is commonly known as an on-device loop.

By executing multiple steps of the model with an on-device loop, you can improve
the performance of your model. This is achieved by creating a ``for`` loop
using ``tf.range`` inside ``tf.function``; AutoGraph will convert this to a
``tf.while_loop`` for you.

For example, the code below creates a custom training loop using an on-device
loop to train a simple model. It uses the syntactical shorthand for infeed
creation and defines an iterator over the outfeed, as described in
:numref:`infeed-simplification`.

.. literalinclude:: targeting_tf2_example3.py
  :language: python
  :linenos:
  :emphasize-lines: 55, 57, 60, 78, 91-93, 95

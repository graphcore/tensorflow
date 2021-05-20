.. _tensorflow2examples:

TensorFlow 2 examples
---------------------

Training on the IPU
~~~~~~~~~~~~~~~~~~~

This example shows how to use the IPU-specific Keras ``Model`` class and the
``IPUStrategyV1`` to train a model using the Keras ``Model.fit()`` method.

The IPU specific changes are highlighted:

* Import the IPU extensions to TensorFlow.
* Create a configuration for the IPU target. To keep things simple, this just
  selects the first available IPU. A configuration can select specific
  IPUs, or groups of IPUs.
* Call the IPU implementation of ``Sequential()``. This has exactly
  the same interface as the standard Keras implementation but uses layers that
  are optimised for the IPU.
* Use the IPU distribution strategy as a context in order to place the training
  code on the configured IPU.

As you can see, there are minimal changes required to get a Keras model running on the
IPU. No changes at all are required to the layers, other than using the IPU-optimised
versions.

.. literalinclude:: example_tf2_fit.py
  :language: python
  :linenos:
  :emphasize-lines: 6,11-13,37,46

Custom training function
~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to use a custom training function with the
``IPUStrategyV1`` and the standard Keras ``Sequential`` class.

.. literalinclude:: example_tf2_custom_training.py
  :language: python
  :linenos:

Pipelined model
~~~~~~~~~~~~~~~

This example shows how to use the IPU-specific Keras pipelined Model
class to train a network.

.. literalinclude:: example_tf2_pipelined_model.py
  :language: python
  :linenos:

This example shows how to use the IPU-specific Keras pipelined Sequential
class to train a network.

.. literalinclude:: example_tf2_pipeline_sequential.py
  :language: python
  :linenos:

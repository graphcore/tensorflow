TensorFlow 2 examples
---------------------

This example shows how to use the IPU-specific Keras ``Model`` class and the
``IPUStrategy`` to train a model using the Keras ``Model.fit()`` method.

.. literalinclude:: example_tf2_model_fit.py

This example shows how to use the IPU-specific Keras ``Sequential`` class and the
``IPUStrategy`` to train a model using the Keras ``Model.fit()`` method.

.. literalinclude:: example_tf2_fit.py

This example shows how to use a custom training function with the
``IPUStrategy`` and the standard Keras ``Sequential`` class.

.. literalinclude:: example_tf2_custom_training.py

This example shows how to use the IPU-specific Keras pipelined model
class to train a network.

.. literalinclude:: example_tf2_pipelined_model.py

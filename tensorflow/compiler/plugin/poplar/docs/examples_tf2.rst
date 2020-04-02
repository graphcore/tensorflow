TensorFlow 2 examples
---------------------

This example shows the Keras API and the ``IPUStrategy`` being used
to train a model using the Keras ``Model.fit()`` method.

.. literalinclude:: example_tf2_fit.py

This example shows the same model being trained using a custom
training function.

.. literalinclude:: example_tf2_custom_training.py

This example shows how to use the IPU specific Keras pipelined model
class to train a network.

.. literalinclude:: example_tf2_pipelined_model.py

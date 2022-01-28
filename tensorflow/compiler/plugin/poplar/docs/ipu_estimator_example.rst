.. _ipu_estimator_example:

Example using IPUEstimator
--------------------------

This example shows how to use the ``IPUEstimator`` to train a simple
CNN on the CIFAR-10 dataset. The XLA compilation is already handled
while using the ``IPUEstimator``, so the ``model_fn`` should not be
manually compiled with ``ipu_compiler``.

.. literalinclude:: ipu_estimator_example.py
  :language: python
  :linenos:

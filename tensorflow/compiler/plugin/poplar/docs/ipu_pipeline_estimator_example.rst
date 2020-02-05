Example using IPUPipelineEstimator
----------------------------------

This example shows how to use the ``IPUPipelineEstimator``
to train a simple CNN on the CIFAR-10 dataset. It can be
compared to the example using the ``IPUEstimator``
(:any:`ipu_estimator_example`) to see the
changes required to add pipelined execution to a model.

.. literalinclude:: ipu_pipeline_estimator_example.py

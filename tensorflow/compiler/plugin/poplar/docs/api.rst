.. _api-section:

Python API
----------

Remember to import the IPU API using:

.. code-block:: python

  from tensorflow.python import ipu

You cannot access the IPU API via the top-level `tensorflow` namespace.
For example, this will not work:

.. code-block:: python

  import tensorflow as tf
  cfg = tf.python.ipu.config.IPUConfig() ...

.. Note automodule:: tensorflow.python.ipu only imports ipu.outlined_function

.. automodule:: tensorflow.python.ipu
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.ipu_strategy
  :members: IPUStrategyV1

.. Note: the headings of the following modules will be included
         as subsection headings. They need to be added before
         introducing subsection headings to group modules together.

.. automodule:: tensorflow.python.ipu.ipu_compiler
  :members:

.. automodule:: tensorflow.python.ipu.scopes
  :members:

.. automodule:: tensorflow.python.ipu.ipu_infeed_queue
  :members:
  :special-members: __init__

.. automodule:: tensorflow.python.ipu.ipu_outfeed_queue
  :members:
  :special-members: __init__

.. General utilities

.. automodule:: tensorflow.python.ipu.utils
  :members:

.. Configuration utilities

.. automodule:: tensorflow.python.ipu.config
  :members:
  :exclude-members: deprecate_config_attribute, deprecate_config_attributes, running_on_ipu_model, IPUConfig, AttributeMetadata

.. autoclass:: tensorflow.python.ipu.config.AttributeMetadata()
  :members:


.. autoclass:: tensorflow.python.ipu.config.IPUConfig

  .. automethod:: tensorflow.python.ipu.config.IPUConfig.get_attribute_metadata()
  .. automethod:: tensorflow.python.ipu.config.IPUConfig.configure_ipu_system()

.. Looping utilities

.. automodule:: tensorflow.python.ipu.loops
  :members:

.. The documentation for this module is incomplete;
   it is also imported from an experimental namespace

.. .. automodule:: tensorflow.python.ipu.popfloat_cast_to_gfloat
..   :members:
..   :imported-members:

.. Distributed training

.. automodule:: tensorflow.python.ipu.ipu_multi_worker_strategy
  :members: IPUMultiWorkerStrategyV1

.. automodule:: tensorflow.python.ipu.horovod
  :members:

.. automodule:: tensorflow.python.ipu.horovod.ipu_horovod_strategy
  :members: IPUHorovodStrategyV1

.. automodule:: tensorflow.python.ipu.horovod.ipu_multi_replica_strategy
  :members: IPUMultiReplicaStrategyV1

.. _datasets-api:

Datasets
^^^^^^^^

.. automodule:: tensorflow.python.ipu.dataset_benchmark
  :members:

.. automodule:: tensorflow.python.ipu.data.ops.dataset_ops
  :members:
  :special-members: __init__
  :imported-members:

.. _estimators-api:

Estimators
^^^^^^^^^^

.. automodule:: tensorflow.python.ipu.ipu_estimator
.. autoclass:: IPUEstimator
  :members:
  :inherited-members:
.. autoclass:: IPUEstimatorSpec
  :members: __new__

.. automodule:: tensorflow.python.ipu.ipu_pipeline_estimator
.. autoclass:: IPUPipelineEstimator
  :members:
  :inherited-members:
.. autoclass:: IPUPipelineEstimatorSpec
  :members: __new__

.. Run configs

.. automodule:: tensorflow.python.ipu.ipu_run_config
  :members:
  :special-members: __new__, __init__

.. Session run hooks

.. automodule:: tensorflow.python.ipu.ipu_session_run_hooks
  :members:
  :special-members: __init__

Keras
^^^^^

.. note::

  `tensorflow.python.ipu.keras.SequentialPipelineModel` has been renamed to
  :py:class:`tensorflow.python.ipu.keras.PipelineSequential` and will be removed
  in a future version.

.. automodule:: tensorflow.python.ipu.keras
  :members: Model, Sequential, PipelineStage, PipelineModel, PipelineSequential
  :imported-members: Model, Sequential, PipelineModel, PipelineStage, PipelineSequential
  :special-members: __init__

.. automodule:: tensorflow.python.ipu.keras.model
  :members: IPUModel, IPUSequential
  :special-members: __init__

.. _keras-layers-api:

Keras layers
^^^^^^^^^^^^

.. note::

  `tensorflow.python.ipu.keras.layers.GRU` is an alias of
  :py:class:`tensorflow.python.ipu.keras.layers.PopnnGRU`

  `tensorflow.python.ipu.keras.layers.LSTM` is an alias of
  :py:class:`tensorflow.python.ipu.keras.layers.PopnnLSTM`


.. automodule:: tensorflow.python.ipu.keras.layers
  :members: Dropout, Embedding, GroupNormalization, InstanceNormalization, LayerNormalization, PopnnGRU, PopnnLSTM, SerialDense, CTCInferenceLayer, CTCPredictionsLayer, RecomputationCheckpoint
  :imported-members: Dropout, Embedding, GroupNormalization, InstanceNormalization, LayerNormalization, PopnnGRU, PopnnLSTM, SerialDense, CTCInferenceLayer, CTCPredictionsLayer, RecomputationCheckpoint

Keras losses
^^^^^^^^^^^^

.. automodule:: tensorflow.python.ipu.keras.losses
  :members: CTCLoss
  :imported-members: CTCLoss

.. _keras-optimizers-api:

Keras optimizers
^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.python.ipu.keras.optimizers
  :members: IpuOptimizer, CrossReplicaOptimizer, MapGradientOptimizer, GradientAccumulationOptimizer
  :imported-members: IpuOptimizer, CrossReplicaOptimizer, MapGradientOptimizer, GradientAccumulationOptimizer

.. _operators-api:

Operators
^^^^^^^^^

It is also possible to access the operators via the
`tensorflow.python.ipu.ops` namespace, for example:
`tensorflow.python.ipu.ops.normalization_ops.group_norm()`.

.. Order alphabetically based on the headings in the source files
.. Put the non-PopLibs ops first

.. automodule:: tensorflow.python.ipu.custom_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.functional_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.image_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.internal_ops
  :members: print_tensor
  :imported-members:

.. automodule:: tensorflow.python.ipu.math_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.pipelining_ops
  :members:
  :special-members: __init__
  :imported-members:
  :exclude-members: Enum, IntEnum

.. The following are all PopLibs ops

.. Popnn

.. automodule:: tensorflow.python.ipu.nn_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.normalization_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.rnn_ops
  :members:
  :special-members: __init__
  :imported-members:

.. Popops

.. automodule:: tensorflow.python.ipu.all_to_all_op
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.cross_replica_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.embedding_ops
  :members:
  :special-members: __init__
  :imported-members:
  :exclude-members: mul, reduce

.. automodule:: tensorflow.python.ipu.reduce_scatter_op
  :members:
  :imported-members:

.. Poprand

.. automodule:: tensorflow.python.ipu.rand_ops
  :members:
  :imported-members:

.. Further ops that come after Pop** alphabetically

.. automodule:: tensorflow.python.ipu.replication_ops
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.slicing_ops
  :members:
  :special-members: __init__
  :imported-members:

.. automodule:: tensorflow.python.ipu.statistics_ops
  :members:
  :special-members: __init__
  :imported-members:

.. automodule:: tensorflow.python.ipu.summary_ops
  :members:
  :imported-members:

.. _optimisers-api:

Optimisers
^^^^^^^^^^

In addition to the `tensorflow.python.ipu.optimizers` namespace, it is also possible to access the optimizer classes via other namespaces, as shown in the following table:

.. table:: Optimizer namespaces
  :width: 100%
  :widths: 45,55

  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  |                  Optimizer                                                                     |                     Alternative namespaces                       |
  +================================================================================================+==================================================================+
  | :class:`~tensorflow.python.ipu.optimizers.CrossReplicaOptimizer`                               | tensorflow.python.ipu.cross_replica_optimizer                    |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.cross_replica_optimizer         |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.CrossReplicaGradientAccumulationOptimizer`           | tensorflow.python.ipu.gradient_accumulation_optimizer            |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.CrossReplicaGradientAccumulationOptimizerV2`         | tensorflow.python.ipu.gradient_accumulation_optimizer            |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizer`                       | tensorflow.python.ipu.gradient_accumulation_optimizer            |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`                     | tensorflow.python.ipu.gradient_accumulation_optimizer            |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.MapGradientOptimizer`                                | tensorflow.python.ipu.map_gradient_optimizer                     |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.map_gradient_optimizer          |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+
  | :class:`~tensorflow.python.ipu.optimizers.ShardedOptimizer`                                    | tensorflow.python.ipu.sharded_optimizer                          |
  |                                                                                                |                                                                  |
  |                                                                                                | tensorflow.python.ipu.optimizers.sharded_optimizer               |
  +------------------------------------------------------------------------------------------------+------------------------------------------------------------------+

.. note:: The `ipu.optimizers` optimizer classes can only be used with subclasses of `tensorflow.compat.v1.train.Optimizer`.

.. automodule:: tensorflow.python.ipu.optimizers
  :members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, IpuOptimizer, MapGradientOptimizer, ShardedOptimizer
  :imported-members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, IpuOptimizer, MapGradientOptimizer, ShardedOptimizer
  :special-members: __init__

.. _sharding-api:

Sharding
^^^^^^^^

.. automodule:: tensorflow.python.ipu.sharding
  :members:

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
  cfg = tf.python.ipu.create_ipu_config(...)

.. Note automodule:: tensorflow.python.ipu only imports ipu.outlined_function

.. automodule:: tensorflow.python.ipu
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.ipu_strategy
  :members: IPUStrategy

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

.. General utiltities

.. automodule:: tensorflow.python.ipu.utils
  :members:

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
  :members: IPUMultiWorkerStrategy

.. automodule:: tensorflow.python.ipu.horovod
  :members:

.. automodule:: tensorflow.python.ipu.horovod.ipu_horovod_strategy
  :members: IPUHorovodStrategy

.. automodule:: tensorflow.python.ipu.horovod.ipu_multi_replica_strategy
  :members: IPUMultiReplicaStrategy

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
  :special-members: __init__

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
  :members: Dropout, Embedding, GroupNormalization, InstanceNormalization, LayerNormalization, PopnnGRU, PopnnLSTM, SerialDense
  :imported-members: Dropout, Embedding, GroupNormalization, InstanceNormalization, LayerNormalization, PopnnGRU, PopnnLSTM, SerialDense

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
  :exclude-members: deprecated_args, Enum, IntEnum

.. The following are all PopLibs ops

.. Popnn

.. tensorflow.python.ipu.nn_ops just contains Gelu

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

.. automodule:: tensorflow.python.ipu.summary_ops
  :members:
  :imported-members:

.. _optimisers-api:

Optimisers
^^^^^^^^^^

In addition to the `tensorflow.python.ipu.optimizers` namespace, it is also possible to access the optimizer classes via other namespaces, as shown in the following table:

+---------------------------------------------+--------------------------------------------------------------------+
|                  Optimizer                  |                       Alternative namespaces                       |
+=============================================+====================================================================+
| CrossReplicaOptimizer                       | - tensorflow.python.ipu.cross_replica_optimizer                    |
|                                             | - tensorflow.python.ipu.optimizers.cross_replica_optimizer         |
+---------------------------------------------+--------------------------------------------------------------------+
| CrossReplicaGradientAccumulationOptimizer   | - tensorflow.python.ipu.gradient_accumulation_optimizer            |
|                                             | - tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
+---------------------------------------------+--------------------------------------------------------------------+
| CrossReplicaGradientAccumulationOptimizerV2 | - tensorflow.python.ipu.gradient_accumulation_optimizer            |
|                                             | - tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
+---------------------------------------------+--------------------------------------------------------------------+
| GradientAccumulationOptimizer               | - tensorflow.python.ipu.gradient_accumulation_optimizer            |
|                                             | - tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
+---------------------------------------------+--------------------------------------------------------------------+
| GradientAccumulationOptimizerV2             | - tensorflow.python.ipu.gradient_accumulation_optimizer            |
|                                             | - tensorflow.python.ipu.optimizers.gradient_accumulation_optimizer |
+---------------------------------------------+--------------------------------------------------------------------+
| MapGradientOptimizer                        | - tensorflow.python.ipu.map_gradient_optimizer                     |
|                                             | - tensorflow.python.ipu.optimizers.map_gradient_optimizer          |
+---------------------------------------------+--------------------------------------------------------------------+
| ShardedOptimizer                            | - tensorflow.python.ipu.sharded_optimizer                          |
|                                             | - tensorflow.python.ipu.optimizers.sharded_optimizer               |
+---------------------------------------------+--------------------------------------------------------------------+

.. automodule:: tensorflow.python.ipu.optimizers
  :members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, MapGradientOptimizer, ShardedOptimizer
  :imported-members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, MapGradientOptimizer, ShardedOptimizer
  :special-members: __init__

.. _sharding-api:

Sharding
^^^^^^^^

.. automodule:: tensorflow.python.ipu.autoshard
  :members:

.. automodule:: tensorflow.python.ipu.sharding
  :members:

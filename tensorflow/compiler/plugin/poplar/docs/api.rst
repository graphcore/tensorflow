.. _api-section:

TensorFlow Python API
---------------------

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

.. note::

  `tensorflow.python.ipu.ipu_strategy.IPUStrategy` is an alias of
  :py:class:`tensorflow.python.ipu.ipu_strategy.IPUStrategyV1`.

.. automodule:: tensorflow.python.ipu.ipu_strategy
  :members: IPUStrategyV1
  :special-members: __init__

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
  :inherited-members:
  :exclude-members: deprecate_config_attribute, deprecate_config_attributes, running_on_ipu_model, IPUConfig

.. autoclass:: tensorflow.python.ipu.config.IPUConfig
  :members:
  :inherited-members:
  :exclude-members: allow_recompute, auto_select_ipus, compilation_poplar_options, convolutions, 
                    device_connection, experimental, floating_point_behaviour, gcl_poplar_options, 
                    io_tiles, ipu_model, matmuls, norms, optimizations, pooling, select_ipus, 
                    selection_order, serialization_output_folder, slices

.. Looping utilities

.. automodule:: tensorflow.python.ipu.loops
  :members:

.. The documentation for this module is incomplete;
   it is also imported from an experimental namespace

.. .. automodule:: tensorflow.python.ipu.popfloat_cast_to_gfloat
..   :members:
..   :imported-members:

.. Distributed training

.. automodule:: tensorflow.python.ipu.distributed
  :members:

.. automodule:: tensorflow.python.ipu.distributed.popdist_strategy
  :members: PopDistStrategy, host_collective_ops
  :special-members: __init__
  :exclude-members: IPUDistributedVariable, IPUMirroredVariable, IPUOnReadPolicy, IPUOnWritePolicy, IPUSyncOnReadVariable, PopDistExtendedV1

.. Serving utilities

.. automodule:: tensorflow.python.ipu.serving
  :members:

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
  :members:
  :inherited-members:
  :exclude-members: export_savedmodel
  

.. automodule:: tensorflow.python.ipu.ipu_pipeline_estimator
  :members:
  :inherited-members:
  :exclude-members: export_savedmodel

.. Run configs

.. automodule:: tensorflow.python.ipu.ipu_run_config
  :members:
  :special-members: __new__, __init__

.. Session run hooks

.. automodule:: tensorflow.python.ipu.ipu_session_run_hooks
  :members:
  :special-members: __init__

.. _operators-api:

Operators
^^^^^^^^^

It is also possible to access the operators via the
`tensorflow.python.ipu.ops` namespace, for example:
`tensorflow.python.ipu.ops.normalization_ops.group_norm()`.

.. Order alphabetically based on the headings in the source files
.. Put the non-PopLibs ops first

.. automodule:: tensorflow.python.ipu.application_compile_op
  :members:
  :imported-members:

.. automodule:: tensorflow.python.ipu.control_flow_ops
  :members:
  :imported-members:

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

.. automodule:: tensorflow.python.ipu.within_replica_ops
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

.. automodule:: tensorflow.python.ipu.embedded_runtime
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

You can configure :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2` and :class:`~tensorflow.python.ipu.optimizers.CrossReplicaGradientAccumulationOptimizerV2`
with an optional reduction method (see :numref:`table_gradient_reduction_options`) defining how to accumulate gradients (see enumerated class :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`).

.. table:: Gradient reduction options
  :name: table_gradient_reduction_options

  +---------------------------+-------------------------------------------------------------------------------+
  | Reduction method          | Behaviour                                                                     |
  +===========================+===============================================================================+
  | `SUM`                     | Sum gradients across the mini-batch.                                          |
  +---------------------------+-------------------------------------------------------------------------------+
  | `MEAN`                    | Sum gradients across the mini-batch after scaling them                        |
  |                           | by (1 / mini-batch-size)                                                      |
  +---------------------------+-------------------------------------------------------------------------------+
  | `RUNNING_MEAN`            | Compute a running mean of gradients across the mini-batch                     |
  |                           | using the expression `acc <- acc*n/(n+1) + grad/(n+1)` for the nth iteration  |
  |                           | within the mini-batch.                                                        |
  +---------------------------+-------------------------------------------------------------------------------+

.. automodule:: tensorflow.python.ipu.gradient_accumulation
  :members: GradientAccumulationReductionMethod
  :imported-members: GradientAccumulationReductionMethod

.. automodule:: tensorflow.python.ipu.optimizers
  :members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, IpuOptimizer, MapGradientOptimizer, ShardedOptimizer
  :imported-members: CrossReplicaOptimizer, CrossReplicaGradientAccumulationOptimizer, CrossReplicaGradientAccumulationOptimizerV2, GradientAccumulationOptimizer, GradientAccumulationOptimizerV2, IpuOptimizer, MapGradientOptimizer, ShardedOptimizer
  :special-members: __init__

.. _sharding-api:

Sharding
^^^^^^^^

.. automodule:: tensorflow.python.ipu.sharding
  :members:

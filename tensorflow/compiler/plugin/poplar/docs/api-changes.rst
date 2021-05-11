API changes
-----------

Release 2.1
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.1.
This may require you to change your code.

Breaking changes
________________

.. warning::

  These will require changes to any code that uses them.

We have removed several items that have been deprecated for at least one
release.

``tensorflow.python.ipu.ops.all_to_all_op.all_gather``

  - The output shape has changed to have the `replication_factor` as the
    outermost instead of innermost dimension, matching the documentation.

``tensorflow.python.ipu.utils``

  - Removed ``report_options`` parameter from ``set_report_options``.

    Use ``graph_options`` and ``execution_options`` parameters instead.

    Only removed for TensorFlow 1.15. Already removed in TensorFlow 2.1.

  - Removed ``allow_stateful_recompute`` parameter from
    ``set_recomputation_options``.

    Pipelining recomputation will recompute all the non-stateful operations when
    recomputation is enabled.

    Only removed for TensorFlow 1.15. Already removed in TensorFlow 2.1.

  - Removed ``num_io_tiles`` from ``set_gcl_options``.

    Use the ``set_io_tile_options`` instead.

    Only removed for TensorFlow 1.15. Already removed in TensorFlow 2.1.

IPUPipelineEstimator change
'''''''''''''''''''''''''''

The definition for ``iterations_per_loop`` has changed. Previously the number of
iterations was defined as the number of weight updates performed. The new
definition is the number of mini-batches consumed, which makes it consistent
with the IPUEstimator when using gradient accumulation.

The argument ``count_gradient_accumulation_as_iterations=True`` was previously
required to use this new definition. That parameter has now been removed and
the new definition is always used.

Autosharding removed
'''''''''''''''''''''''

Autosharding has been removed. You should now use alternative execution modes
such as pipelining instead.

IPU config change
'''''''''''''''''

The ``disable_graph_convolution_caching`` parameter for ``create_ipu_config``
(from ``tensorflow.python.ipu.utils``) has been removed.

The ``disable_graph_outlining`` parameter must be used instead.

IPU Keras changes [TensorFlow 2]
''''''''''''''''''''''''''''''''

The ``SequentialPipelineModel`` alias for ``PipelineSequential`` has been
removed.

In the constructors of ``ipu.keras.Model`` and ``ipu.keras.Sequential``,
the alias ``accumulation_count`` for the ``gradient_accumulation_count``
parameter has been removed.

Similarly, the alias ``accumulation_dtype`` for ``gradient_accumulation_dtype``
has been removed.

Non-breaking changes
____________________

These changes are recommended.

Recompute suggestions deprecated
''''''''''''''''''''''''''''''''

The ``recompute`` and ``block_recompute`` utility ops have been deprecated and will be removed
in release 2.2. Automatic recomputation of casts will remain.


IPUOutfeedQueue/IPUOutfeedQueue replication_factor deprecated
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The ``replication_factor`` argument of ``IPUInfeedQueue``and ``IPUOutfeedQueue`` has been deprecated
and will be removed in release 2.2. The replication factor is now automatically set based on the model
being executed.

CTC loss ops deprecated
''''''''''''''''''''''''''''''''

The ``ctc_loss`` and ``ctc_loss_with_logits`` ops from ``ipu.ops.nn_ops`` have been deprecated and
will be removed in release 2.2. They have been superseeded by ``ctc_loss_v2`` and
``ctc_loss_with_log_probs``.

Changes to configuration API
''''''''''''''''''''''''''''

``tensorflow.python.ipu.create_ipu_config``

  - The ``scheduler_selection`` argument has been changed from taking a string
    value to a value from the new
    ``tensorflow.python.ipu.utils.SchedulingAlgorithm`` enumeration. Strings can
    still be passed and an attempt will be made to convert them to the new
    enumeration. The mapping is as follows:
      - "": ``SchedulingAlgorithm.CHOOSE_BEST``
      - "Clustering": ``SchedulingAlgorithm.CLUSTERING``
      - "PostOrder": ``SchedulingAlgorithm.POST_ORDER``
      - "LookAhead": ``SchedulingAlgorithm.LOOK_AHEAD``
      - "ShortestPath": ``SchedulingAlgorithm.SHORTEST_PATH``

``tensorflow.python.ipu.set_ipu_connection_type``

  - The ``ipu_version`` argument has been changed from taking an integer to
    taking a string, in line with the ``ipu_model_version`` argument in
    ``tensorflow.python.ipu.utils.set_ipu_model_options``. Integers can still
    be passed and an attempt will be made to convert them to strings. The
    mapping is as follows:
      - 1: "ipu1"
      - 2: "ipu2"

``tensorflow.python.ipu.set_optimization_options``

  - The ``merge_remote_buffers`` argument has been changed from taking a boolean
    or ``None`` value to a value from the new
    ``tensorflow.python.ipu.utils.MergeRemoteBuffersBehaviour`` enumeration.
    Booleans or ``None`` can still be passed and an attempt will be made to
    convert them to the new enumeration. The mapping is as follows:
      - ``True``: ``MergeRemoteBuffersBehaviour.MERGE``
      - ``False``: ``MergeRemoteBuffersBehaviour.NO_MERGING``
      - ``None``: ``MergeRemoteBuffersBehaviour.IF_BENEFICIAL``
    The default value has also changed - the default is now
    ``MergeRemoteBuffersBehaviour.IF_BENEFICIAL``, while before it was
    ``False`` (disabled by default).

The following functions and classes have moved to a new file
``tensorflow/python/ipu/config.py``, but they are still accessible from
``tensorflow/python/ipu/utils.py`` as usual:
  - ``configure_ipu_system``
  - ``get_ipu_config``
  - ``SelectionOrder``
  - ``ExecutionProfileType``
  - ``DeviceConnectionType``
  - ``MergeRemoteBuffersBehaviour``
  - ``SchedulingAlgorithm``
  - ``KeyId``
  - ``VerificationOptions``

Support for grouped collectives
'''''''''''''''''''''''''''''''

``tensorflow.python.ipu.ops.all_to_all_op.all_gather``
``tensorflow.python.ipu.ops.reduce_scatter_op.reduce_scatter``

  - The ``replication_factor`` can now be set to a value smaller than the
    total number of replicas in the model, in which case the collective
    operation will be performed within groups of the given size.

``tensorflow.python.ipu.ops.cross_replica_ops.cross_replica_sum``

  - A new optional argument ``replica_group_size`` is added for specifying
    the number of replicas in each collective group. If not specified, there
    is a single group containing all the replicas.

Release 2.0
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.0.
This may require you to change your code.

Breaking changes
________________

.. warning::

  These will require changes to any code that uses them.

We have removed several items that have been deprecated for at least one
release.

``tensorflow.python.ipu.ipu_outfeed_queue``

  - Removed ``outfeed_all`` parameter from ``IPUOutfeedQueue``.

    Use ``outfeed_mode`` parameter instead.

``tensorflow.python.ipu.ipu_pipeline_estimator``

  - Removed ``pipeline_depth`` parameter from
    ``IPUPipelineEstimatorSpec``.

    Use ``gradient_accumulation_count parameter instead``.

``tensorflow.python.ipu.utils``

  - Removed ``retain_control_dependencies`` parameter from
    ``create_ipu_config``.

    Only removed in TensorFlow 2.1.

  - Removed ``max_cross_replica_sum_buffer_size``, and
    ``max_inter_ipu_copies_buffer_size`` parameters from
    ``create_ipu_config``.

    Use ``set_optimization_options`` instead.

  - Removed ``report_options`` parameter from ``set_report_options``.

    Use ``graph_options`` and ``execution_options`` parameters instead.

  - Removed ``allow_stateful_recompute`` parameter from
    ``set_recomputation_options``.

    Pipelining recomputation will recompute all the non-stateful operations when
    recomputation is enabled.

  - Removed ``num_io_tiles`` from ``set_gcl_options``.

    Use the ``set_io_tile_options`` instead.

``tensorflow.python.ipu.ops.embedding_ops.embedding_lookup``

  - Removed ``one_hot_threshold`` and ``min_encoding_size`` parameters
    from ``embedding_lookup``.

  - Removed ``count`` parameter from ``HostEmbeddingScope.lookup``.

``tensorflow.python.ipu.ops.functional_ops``

  - Removed ``function``.

    Use ``outlined_function`` instead.

``tensorflow.python.ipu.ops.normalization_ops``

  - Removed ``reduction_axes`` parameter from ``group_norm``,
    ``layer_norm``, and ``instance_norm``.

``tensorflow.python.ipu.ops.pipelining_ops``

  - Removed ``pipeline_depth`` parameter from ``pipeline``.

    Use ``gradient_accumulation_count`` instead.

``tensorflow.python.ipu.ops.rnn_ops``

  - Removed support for passing a tuple as the ``initial_state``
    argument for ``PopnnLSTM.call``.

    This must be an ``LSTMStateTuple`` now.

The following deprecated namespace has been removed:

  * ``tensorflow.python.ipu.ipu_optimizer``

  Use the ``tensorflow.python.ipu.optimizers`` namespace instead.



Non-breaking changes
____________________

These changes are recommended.

IPUPipelineEstimator change
'''''''''''''''''''''''''''

The definition for ``iterations_per_loop`` has changed. Previously the number of
iterations was defined as the number of weight updates performed. The new
definition is the number of mini-batches consumed, which makes it consistent
with the IPUEstimator when using gradient accumulation. The old definition is
still used by default, but it will be removed in a future release.

Use the argument ``count_gradient_accumulation_as_iterations=True`` to use the
new definition.

Autosharding deprecated
'''''''''''''''''''''''

Autosharding has been deprecated, and will be removed in a future release. You
should now use alternative execution modes such as pipelining instead.

IPU config change
'''''''''''''''''

The ``disable_graph_convolution_caching`` parameter for ``create_ipu_config``
(from ``tensorflow.python.ipu.utils``) has been deprecated as it has no effect.
It will be removed in a future release.

The ``disable_graph_outlining`` parameter should be used instead.

IPU Keras changes [TensorFlow 2]
''''''''''''''''''''''''''''''''

``SequentialPipelineModel`` has been renamed to ``PipelineSequential`` for
consistency with its ``Model`` counterpart. The old name can still be used, but
is deprecated and will be removed in a future release.

The ``accumulation_count`` argument in the constructors of the
``ipu.keras.Model`` and ``ipu.keras.Sequential`` has been renamed to
``gradient_accumulation_count`` to be consistent with the rest of the code base.
The old name can still be used, but is deprecated and will be removed in a
future release.

Similarly, ``accumulation_dtype`` has been renamed to ``gradient_accumulation_dtype``.

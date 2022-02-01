API changes
-----------

Release 2.5
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.5.
This may require you to change your code.

Non-breaking changes
____________________

.. _layers-moved-to-addons:

Deprecated layers
'''''''''''''''''

TensorFlow layers from `tensorflow.python.ipu.ops.rnn_ops` have been moved to
the `ipu_tensorflow_addons.layers` namespace in IPU TensorFlow Addons.

The layers have been deprecated in TensorFlow and will be removed in a future
release.

The table below lists all of the deprecated layers, and their new locations:

+----------------------------------------------------+-------------------------------------------------------+
| **TensorFlow**                                     | **IPU TensorFlow Addons**                             |
+====================================================+=======================================================+
| tensorflow.python.ipu.ops.rnn_ops.PopnnAUGRU       | ipu_tensorflow_addons.layers.rnn_ops.PopnnAUGRU       |
| tensorflow.python.ipu.ops.rnn_ops.PopnnDynamicGRU  | ipu_tensorflow_addons.layers.rnn_ops.PopnnDynamicGRU  |
| tensorflow.python.ipu.ops.rnn_ops.PopnnDynamicLSTM | ipu_tensorflow_addons.layers.rnn_ops.PopnnDynamicLSTM |
| tensorflow.python.ipu.ops.rnn_ops.PopnnGRU         | ipu_tensorflow_addons.layers.rnn_ops.PopnnGRU         |
| tensorflow.python.ipu.ops.rnn_ops.PopnnLSTM        | ipu_tensorflow_addons.layers.rnn_ops.PopnnLSTM        |
+----------------------------------------------------+-------------------------------------------------------+

RNN available_memory_proportion_fwd/available_memory_proportion_bwd deprecated
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The ``available_memory_proportion_fwd`` and ``available_memory_proportion_bwd`` arguments have been deprecated and will be removed from the following layers in a future release:

  - tensorflow.python.ipu.ops.rnn_ops.PopnnLSTM
  - tensorflow.python.ipu.ops.rnn_ops.PopnnDynamicLSTM
  - tensorflow.python.ipu.ops.rnn_ops.PopnnGRU
  - tensorflow.python.ipu.ops.rnn_ops.PopnnDynamicGRU
  - tensorflow.python.ipu.ops.rnn_ops.PopnnAUGRU

These values are now set using the ``'availableMemoryProportion'`` key of the ``options`` and ``options_bwd`` arguments correspondingly.

Release 2.4
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.4.
This may require you to change your code.

Breaking changes
________________

.. warning::

  These will require changes to any code that uses them.

Summary ops
'''''''''''

The following items related to summary ops have been deprecated, **are no longer
functional** and will be removed in a future release. To profile IPU programs,
use the PopVision suite of analysis tools. Trying to use these items will raise
a `NotImplementedError`:

  - `tensorflow.python.ipu.ops.summary_ops`:

    - `ipu_compile_summary`

  - `IPUEstimator`:

    - The `compile_summary` argument to :py:class:`~tensorflow.python.ipu.ipu_run_config.IPURunConfig`
    - Passing a `IPURunConfig` with `compile_summary` set to `True` to an `IPUEstimator`


Removal of deprecated members
'''''''''''''''''''''''''''''

The following have been removed, as they were deprecated in a previous release:

  - The following ``TF_POPLAR_FLAGS``:

    - ``dump_text_reports_to_stdio``
    - ``add_all_reduce_copies``
    - ``force_replicated_mode``
    - ``save_oom_profiler``


  - The following constructor arguments for :py:class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`:

    - ``replication_factor``
    - ``data_to_prefetch``
    - ``feed_name``


  - The following constructor arguments for :py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`:

    - ``replication_factor``
    - ``io_batch_size``
    - ``feed_name``


  - The following constructor arguments for :py:class:`~tensorflow.python.ipu.ipu_session_run_hooks.IPULoggingTensorHook`:

    - ``replication_factor``
    - ``feed_name``


  - The following functions from `tensorflow.python.ipu.utils`:

    - ``create_ipu_config``
    - ``set_serialization_options``
    - ``set_optimization_options``
    - ``set_norm_options``
    - ``set_compilation_options``
    - ``set_convolution_options``
    - ``set_matmul_options``
    - ``set_pooling_options``
    - ``set_report_options``
    - ``set_ipu_model_options``
    - ``set_recomputation_options``
    - ``set_floating_point_behaviour_options``
    - ``set_io_tile_options``
    - ``set_gcl_options``
    - ``auto_select_ipus``
    - ``select_ipus``
    - ``set_ipu_connection_type``
    - ``set_experimental_multi_replica_distribution_options``
    - ``extract_compile_reports``
    - ``extract_poplar_serialized_graphs``
    - ``extract_execute_reports``


  - The following functions from `tensorflow.python.ipu.ops.nn_ops`:

    - ``ctc_loss``
    - ``ctc_loss_with_logits``

  - The following functions from `tensorflow.python.ipu.ops.internal_ops`:

    - ``recompute``
    - ``block_recompute``


Additionally, the documentation section on profiling through the deprecated
TensorFlow profiling APIs has been removed and an ``IpuOptions`` configuration
protobuf can no longer be passed to the
:py:class:`~tensorflow.python.ipu.ipu_run_config.IPURunConfig` constructor.


Non-breaking changes
____________________

  - The following functions from `tensorflow.python.ipu.utils` are now
    considered internal-only tools and have correspondingly been moved to
    `tensorflow.compiler.plugin.poplar.tests.test_utils`. They can still be
    accessed from their previous location, but not in future releases:

    - ``extract_all_events``
    - ``extract_all_strings_from_event_trace``
    - ``extract_all_types_from_event_trace``

  - 'IPUConfig.floating_point_behaviour.esr' - Assigning a bool value is
    deprecated and will not be supported in a future release.
    :py:class:`~tensorflow.python.ipu.config.StochasticRoundingBehaviour` should
    be used instead.
  - `ipu_multi_replica_strategy.IPUMultiReplicaStrategy` has been renamed to
    `popdist_strategy.PopDistStrategy`. Using `ipu_multi_replica_strategy.IPUMultiReplicaStrategy`
    will trigger a deprecation warning.
  - `IPUMultiWorkerStrategy` is deprecated. Using `IPUMultiWorkerStrategy`
    will trigger a deprecation warning.
  - The flag `save_interval_report` from `TF_POPLAR_FLAGS` is now deprecated. Please
    use LIBPVA instead.

Release 2.3
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.3.
This may require you to change your code.

Breaking changes
________________

.. warning::

  These will require changes to any code that uses them.

Custom user op metadata interface updates
'''''''''''''''''''''''''''''''''''''''''

The metadata interface for custom user ops has been updated with an additional parameter.

Existing user ops must update their `custom_op_api_level` value to `5` and update their
metadata function to match the following signature

.. code-block:: cpp
  :linenos:

  void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::vector<std::int64_t>& replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise, bool& is_stateless, bool& is_hashable,
    std::uint32_t num_inputs);

The verified transfers feature has been removed
'''''''''''''''''''''''''''''''''''''''''''''''

The following functions from `tensorflow.python.ipu.utils` have been removed:

  - `set_transfer_options`
  - `set_verification_options`

The following classes from `tensorflow.python.ipu.config` have been removed:

  - `KeyId`
  - `VerificationOptions`


Non-breaking changes
____________________

  - `IPUConfig.optimizations.enable_fast_math` has been moved to `IPUConfig.optimizations.math.fast`

Release 2.2
~~~~~~~~~~~

The following changes have been made to the TensorFlow API in the Poplar SDK version 2.2.
This may require you to change your code.

Breaking changes
________________

.. warning::

  These will require changes to any code that uses them.

C++ Poplar TensorFlow libraries are private by default
''''''''''''''''''''''''''''''''''''''''''''''''''''''

Users interested in targeting the IPU from C++ are required to use the new ipu_config library.
We've made most C++ libraries produced as part of the Poplar backend private, so dependencies
on ``poplar:driver`` and other libraries will no longer be valid and should be replaced with a dependency
to ``//tensorflow/compiler/plugin/poplar:ipu_config``. This library provides a public interface for configuring
IPUs in C++, all other operations should use the standard TensorFlow C++ API. No other Poplar TensorFlow libraries should be
directly depended on.


Reports removed from ipu events
'''''''''''''''''''''''''''''''''

Following the exclusion of profiling options from the :ref:`new-configuration-api`, reports have
been removed from IPU events. The following functions from `tensorflow.python.ipu.utils` have been
deprecated and now return blank lists:

  - extract_compile_reports
  - extract_poplar_serialized_graphs
  - extract_execute_reports

See the :ref:`new-configuration-api` changes for information on profiling TensorFlow programs using
the profiling tools available in the SDK.


Non-breaking changes
____________________

These changes are recommended.


IPULoggingTensorHook replication_factor deprecated
''''''''''''''''''''''''''''''''''''''''''''''''''

The ``replication_factor`` argument of ``IPULoggingTensorHook`` will be removed
in release 2.3. The replication factor is now automatically set based on the
model being executed.


IPUInfeedQueue/IPUOutfeedQueue/IPULoggingTensorHook feed_name deprecated
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The ``feed_name`` argument of ``IPUInfeedQueue``, ``IPUOutfeedQueue`` and
``IPULoggingTensorHook`` has been deprecated and will be removed in release 2.3.
The ``feed_name`` is now generated automatically internally.

Change of output location for profiling information
'''''''''''''''''''''''''''''''''''''''''''''''''''

By default the profile information (``profile.pop`` & ``frameworks.json``) will now be output to a
subdirectory of the Poplar ``autoReport.directory``. If ``autoReport.directory`` is not set, it will be output to
a subdirectory of the current working directory. This change means that mutliple
profiles can be captured for a single model, if it is separated into different Poplar graphs.

The subdirectories are created using the following format ``tf_report__<iso_date>__<pid>``
and the cluster name can be read from the ``frameworks.json`` file in each subdirectory.

IPU Keras Layers deprecation in TensorFlow 1.15
'''''''''''''''''''''''''''''''''''''''''''''''

IPU Keras layers (``AssumeEqualAcrossReplicas``, ``Dropout``, ``Embedding``,
``GroupNormalization``, ``InstanceNormalization``, ``LayerNormalization``,
``RecomputationCheckpoint``, ``PopnnLSTM`` and ``PopnnGRU``) are deprecated and
will be removed in the next release. If you require Keras support please migrate
your model to TensorFlow 2 which has full Keras support for IPU.

Warning when epsilon value is too low
'''''''''''''''''''''''''''''''''''''

When the epsilon value given to ``instance_norm``, ``layer_norm`` or ``group_norm`` is less than 1.53e-5, a warning
will show on the screen that explains the potential dangers and suggests to increase it.

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

Old IPU option configuration API changes
''''''''''''''''''''''''''''''''''''''''

.. note::
  These are changes to the old option configuration API. A new option
  configuration API has been introduced in this release and the old API is
  being deprecated. For more information, please see :ref:`new-configuration-api`.

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


IPUInfeedQueue/IPUOutfeedQueue replication_factor deprecated
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The ``replication_factor`` argument of ``IPUInfeedQueue`` and ``IPUOutfeedQueue`` has been deprecated
and will be removed in release 2.2. The replication factor is now automatically set based on the model
being executed.


IPUInfeedQueue data_to_prefetch deprecated
''''''''''''''''''''''''''''''''''''''''''

The ``data_to_prefetch`` argument of ``IPUInfeedQueue`` has been deprecated and
will be removed in release 2.2. It is recommended to use the ``prefetch_depth``
argument instead.


IPUOutfeedQueue data_to_prefetch deprecated
'''''''''''''''''''''''''''''''''''''''''''

The ``io_batch_size`` argument of ``IPUOutfeedQueue`` has been deprecated and
will be removed in release 2.2. It is recommended to either manually accumulate
results or use ``accumulate_outfeed`` when using pipelining.

CTC loss ops deprecated
'''''''''''''''''''''''

The ``ctc_loss`` and ``ctc_loss_with_logits`` ops from ``ipu.ops.nn_ops`` have been deprecated and
will be removed in release 2.2. They have been superseeded by ``ctc_loss_v2`` and
``ctc_loss_with_log_probs``.

.. _new-configuration-api:

New configuration API
'''''''''''''''''''''

A new API for configuring the IPU system has been added which is replacing the
current API. The new API consists of a single class called
``IPUConfig`` with a hierarchical organisation of options as attributes.
You can set options by assigning values to the attributes of an instance of this
class. The class includes some usability features which should make the process
of configuring the IPU system easier and with no hidden pitfalls. For more
information about the new API, see :ref:`configuring-section`.

.. warning::

  The new ``IPUConfig`` API does not include the profiling options in the
  former configuration API, such as ``profiling``, ``profile_execution``,
  ``report_every_nth_execution`` etc.
  To profile a TensorFlow program, you should instead use the suite of profiling
  tools that have been added to the SDK. For general advice on how to enable
  profiling, refer to the :ref:`Capturing IPU Reports <report_capture>` chapter
  in the PopVision User Guide. To parse profiles, use the
  :std:doc:`PopVision Analysis Python API chapter <pva-python>` or :std:doc:`PopVision Analysis C++ API chapter <pva>`
  in the Poplar and PopLibs API Reference. To enable time-based profiling of
  events, see the :ref:`Capturing Execution Information <{HelpTopic.CapturingData}>`
  chapter of the PopVision User Guide.

  Note that any Poplar engine options mentioned in the above guides can be
  passed to the :ref:`compilation_poplar_options <compilation_poplar_options>`
  ``IPUConfig`` option, so it is not impossible to enable profiling using the
  new configuration API *directly*, but it is not advised, as environment
  variables will overwrite any values set this way.

.. warning::

  The new ``IPUConfig`` API does not support verified transfers. This means the
  verified transfers feature will be removed when the old API is removed.

The new ``IPUConfig`` class is in a new namespace
``tensorflow.python.ipu.config``. Multiple functions and classes have moved from
``tensorflow.python.ipu.utils`` to the ``config`` namespace:
  - ``configure_ipu_system()``
  - ``get_ipu_config()``
  - ``SelectionOrder``
  - ``ExecutionProfileType``
  - ``DeviceConnectionType``
They can still be accessed from ``tensorflow.python.ipu.utils`` - along with
``IPUConfig`` - and there are currently no plans to remove this additional
access route.

To help in converting from the old configuration API to the new API, the
following table shows which attribute of ``IPUConfig`` each function argument in
the old API corresponds to and how:

.. table:: Configuration API conversion
  :width: 100%

  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | Old API function                                                                            | Function argument                                          | ``IPUConfig`` attribute equivalent                                                                                                 |
  +=============================================================================================+============================================================+====================================================================================================================================+
  | :py:func:`~tensorflow.python.ipu.utils.create_ipu_config`                                   | ``profiling``                                              | Not supported in IPUConfig. Use the autoReport.outputGraphProfile or autoReport.all Poplar engine options.                         |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``enable_ipu_events``                                      | Not supported in IPUConfig. Use the PopVision System Analyser to inspect compilation, transfer and execution events.               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``use_poplar_text_report``                                 | Not supported in IPUConfig. Use the PopVision Graph Analyser for manual inspection of reports.                                     |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``use_poplar_cbor_report``                                 | Not supported in IPUConfig. You can set the profiler.format Poplar engine option to the *deprecated* "v1" value for CBOR reports.  |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``profile_execution``                                      | Not supported in IPUConfig. Use the autoReport.all and debug.computeInstrumentationLevel Poplar engine options.                    |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``enable_poplar_serialized_graph``                         | Not supported in IPUConfig. Use the autoReport.outputSerializedGraph or autoReport.all Poplar engine options instead.              |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``report_every_nth_execution``                             | Not supported in IPUConfig. This feature will be removed when the former configuration API is removed.                             |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_report_size``                                        | Not supported in IPUConfig. The Poplar profiling format's storage size has been significantly improved.                            |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``report_directory``                                       | Not supported in IPUConfig. To make module profiling files go into their own sub-directories, do **not** set autoReport.directory. |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``scheduler_selection``                                    | :ref:`scheduling.algorithm <scheduling.algorithm>` [#]_                                                                            |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``always_rearrange_copies_on_the_host``                    | :ref:`experimental.always_rearrange_copies_on_the_host <experimental.always_rearrange_copies_on_the_host>`                         |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``merge_infeed_io_copies``                                 | :ref:`optimizations.merge_infeed_io_copies <optimizations.merge_infeed_io_copies>`                                                 |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``disable_graph_outlining``                                | :ref:`optimizations.enable_graph_outlining <optimizations.enable_graph_outlining>` [#]_                                            |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_scheduler_lookahead_depth``                          | :ref:`scheduling.maximum_scheduler_lookahead_depth <scheduling.maximum_scheduler_lookahead_depth>`                                 |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_scheduler_search_space_size``                        | :ref:`scheduling.maximum_scheduler_search_space_size <scheduling.maximum_scheduler_search_space_size>`                             |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``prefetch_data_streams``                                  | :ref:`optimizations.prefetch_data_streams <optimizations.prefetch_data_streams>`                                                   |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``selection_order``                                        | :ref:`selection_order <selection_order>`                                                                                           |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``enable_experimental_remote_buffer_embedding``            | :ref:`experimental.enable_remote_buffer_embedding <experimental.enable_remote_buffer_embedding>`                                   |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_serialization_options`                           | ``output_folder``                                          | :ref:`serialization_output_folder <serialization_output_folder>`                                                                   |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_optimization_options`                            | ``combine_embedding_lookups``                              | :ref:`optimizations.combine_embedding_lookups <optimizations.combine_embedding_lookups>`                                           |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``combine_matmuls``                                        | :ref:`optimizations.combine_matmuls <optimizations.combine_matmuls>`                                                               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_cross_replica_sum_buffer_size``                      | :ref:`optimizations.maximum_cross_replica_sum_buffer_size <optimizations.maximum_cross_replica_sum_buffer_size>`                   |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_reduce_scatter_buffer_size``                         | :ref:`optimizations.maximum_reduce_scatter_buffer_size <optimizations.maximum_reduce_scatter_buffer_size>`                         |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_inter_ipu_copies_buffer_size``                       | :ref:`optimizations.maximum_inter_ipu_copies_buffer_size <optimizations.maximum_inter_ipu_copies_buffer_size>`                     |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``max_send_recv_cluster_size``                             | :ref:`optimizations.maximum_send_recv_cluster_size <optimizations.maximum_send_recv_cluster_size>`                                 |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``minimum_remote_tensor_size``                             | :ref:`optimizations.minimum_remote_tensor_size <optimizations.minimum_remote_tensor_size>`                                         |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``merge_remote_buffers``                                   | :ref:`optimizations.merge_remote_buffers <optimizations.merge_remote_buffers>` [#]_                                                |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``gather_simplifier``                                      | :ref:`optimizations.enable_gather_simplifier <optimizations.enable_gather_simplifier>`                                             |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``triangular_solve_expander_block_size``                   | :ref:`optimizations.triangular_solve_expander_block_size <optimizations.triangular_solve_expander_block_size>`                     |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``cholesky_block_size``                                    | :ref:`optimizations.cholesky_block_size <optimizations.cholesky_block_size>`                                                       |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``enable_fast_math``                                       | :ref:`optimizations.enable_fast_math <optimizations.enable_fast_math>`                                                             |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_norm_options`                                    | ``use_stable_statistics``                                  | :ref:`norms.use_stable_statistics <norms.use_stable_statistics>`                                                                   |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``experimental_distributed_batch_norm_replica_group_size`` | :ref:`norms.experimental.distributed_batch_norm_replica_group_size <norms.experimental.distributed_batch_norm_replica_group_size>` |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_transfer_options`                                | ``use_verified_transfers``                                 | Not supported with IPUConfig. Verified transfers will be removed when the former configuration API is removed.                     |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+                                                                                                                                    |
  | :py:func:`~tensorflow.python.ipu.utils.set_verification_options`                            | ``verification_options``                                   |                                                                                                                                    |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_compilation_options`                             | ``compilation_options`` [7]_                               | :ref:`compilation_poplar_options <compilation_poplar_options>`                                                                     |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_convolution_options`                             | ``convolution_options`` [7]_                               | :ref:`convolutions.poplar_options <convolutions.poplar_options>`                                                                   |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_matmul_options`                                  | ``matmul_options`` [7]_                                    | :ref:`matmuls.poplar_options <matmuls.poplar_options>`                                                                             |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``clear_pass_type``                                        | :ref:`matmuls.clear_pass_type <matmuls.clear_pass_type>`                                                                           |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_pooling_options`                                 | ``pooling_options`` [7]_                                   | :ref:`pooling.poplar_options <pooling.poplar_options>`                                                                             |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_report_options`                                  | ``graph_options``                                          | Not supported in IPUConfig. All graph report options have equivalents in the PopVision Graph Analyser or PopVision Analysis APIs   |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``execution_options``                                      | Not supported in IPUConfig. All execution report options have equivalents in the PopVision Graph Analyser                          |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_ipu_model_options`                               | ``compile_ipu_code``                                       | :ref:`ipu_model.compile_ipu_code <ipu_model.compile_ipu_code>`                                                                     |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``tiles_per_ipu``                                          | :ref:`ipu_model.tiles_per_ipu <ipu_model.tiles_per_ipu>`                                                                           |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``ipu_model_version``                                      | :ref:`ipu_model.version <ipu_model.version>`                                                                                       |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_recomputation_options` [#]_                      | ``allow_recompute``                                        | :ref:`allow_recompute <allow_recompute>`                                                                                           |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_floating_point_behaviour_options` [#]_           | ``inv``                                                    | :ref:`floating_point_behaviour.inv <floating_point_behaviour.inv>`                                                                 |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``div0``                                                   | :ref:`floating_point_behaviour.div0 <floating_point_behaviour.div0>`                                                               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``oflo``                                                   | :ref:`floating_point_behaviour.oflo <floating_point_behaviour.oflo>`                                                               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``esr``                                                    | :ref:`floating_point_behaviour.esr <floating_point_behaviour.esr>`                                                                 |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``nanoo``                                                  | :ref:`floating_point_behaviour.nanoo <floating_point_behaviour.nanoo>`                                                             |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_io_tile_options`                                 | ``num_io_tiles``                                           | :ref:`io_tiles.num_io_tiles <io_tiles.num_io_tiles>`                                                                               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``place_ops_on_io_tiles``                                  | :ref:`io_tiles.place_ops_on_io_tiles <io_tiles.place_ops_on_io_tiles>`                                                             |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``io_tile_available_memory_proportion``                    | :ref:`io_tiles.available_memory_proportion <io_tiles.available_memory_proportion>`                                                 |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_gcl_options`                                     | ``gcl_options`` [7]_                                       | :ref:`gcl_poplar_options <gcl_poplar_options>`                                                                                     |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.auto_select_ipus`                                    | ``num_ipus``                                               | :ref:`auto_select_ipus <auto_select_ipus>`                                                                                         |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.select_ipus`                                         | ``indices``                                                | :ref:`select_ipus <select_ipus>`                                                                                                   |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_ipu_connection_type`                             | ``connection_type``                                        | :ref:`device_connection.type <device_connection.type>` [#]_                                                                        |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``ipu_version``                                            | :ref:`device_connection.version <device_connection.version>`                                                                       |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``enable_remote_buffers``                                  | :ref:`device_connection.enable_remote_buffers <device_connection.enable_remote_buffers>`                                           |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  | :py:func:`~tensorflow.python.ipu.utils.set_experimental_multi_replica_distribution_options` | ``process_count``                                          | :ref:`experimental.multi_replica_distribution.process_count <experimental.multi_replica_distribution.process_count>`               |
  |                                                                                             +------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
  |                                                                                             | ``process_index``                                          | :ref:`experimental.multi_replica_distribution.process_index <experimental.multi_replica_distribution.process_index>`               |
  +---------------------------------------------------------------------------------------------+------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------+

.. [#] ``IPUConfig.scheduling.algorithm`` takes a value from the new
        :py:class:`~tensorflow.python.ipu.config.SchedulingAlgorithm`
        enumeration, whereas the former configuration API took a string. The
        old string values map to the enumeration as follows:

        - "": ``SchedulingAlgorithm.CHOOSE_BEST``
        - "Clustering": ``SchedulingAlgorithm.CLUSTERING``
        - "PostOrder": ``SchedulingAlgorithm.POST_ORDER``
        - "LookAhead": ``SchedulingAlgorithm.LOOK_AHEAD``
        - "ShortestPath": ``SchedulingAlgorithm.SHORTEST_PATH``

.. [#] ``IPUConfig.optimizations.enable_graph_outlining`` takes a boolean value
       that specifies whether or not graph outlining should be enabled. A value
       of True means that graph outlining is enabled. This is different to the
       old configuration API, which took a boolean value that specifies whether
       or not graph outlining should be **disabled**. Therefore, you should
       invert the boolean you gave to the old configuration API when passing it
       to an IPUConfig.

.. [#] ``IPUConfig.optimizations.merge_remote_buffers`` takes a value from the
        new
        :py:class:`~tensorflow.python.ipu.config.MergeRemoteBuffersBehaviour`
        enumeration, whereas the former configuration API took a boolean or
        None value. The old values map to the enumeration as follows:

        - ``True``: ``MergeRemoteBuffersBehaviour.MERGE``
        - ``False``: ``MergeRemoteBuffersBehaviour.NO_MERGING``
        - ``None``: ``MergeRemoteBuffersBehaviour.IF_BENEFICIAL``
        The ``IPUConfig`` also sets the default value to ``IF_BENEFICIAL``,
        whereas the old configuration API sets the default value to
        ``NO_MERGING``.

.. [#] In the old configuration API, a call to ``set_recomputation_options``
       would make the ``allow_recompute`` argument True by default, therefore
       merely calling ``set_recomputation_options(opts)`` would turn
       recomputation on. Please bear this in mind when moving to ``IPUConfig``.

.. [#] In the old configuration API, a call to
       ``set_floating_point_behaviour_options`` would make all of the arguments
       True by default, therefore merely calling
       ``set_floating_point_behaviour_options(opts)`` would turn all of ``inv``,
       ``oflo``, ``nanoo``, ``div0`` and ``esr`` on. Please bear this in mind
       when moving to ``IPUConfig``. Note that there is the
       :ref:`floating_point_behaviour.set_all <floating_point_behaviour.set_all>`
       option to unconditionally set all of these options on provided for
       convenience.

.. [#] ``IPUConfig.device_connection.version`` takes a string, whereas the
        former configuration API took an integer. The old values map to the
        string values as follows:

        - 1: "ipu1"
        - 2: "ipu2"

.. [7] In the old configuration API, all options dictionaries are accumulative
       each time their function is called. For example, doing:

       .. code-block:: python

         opts = set_compilation_options(opts, {"option1": "true"})
         ...
         opts = set_compilation_options(opts, {"option2": "5"})

       would mean that Poplar compilation is given both options
       ``{"option1": "true", "option2": "5"}``.

       In the ``IPUConfig`` API, this is not the case, as these options
       dictionaries are like any other Python dictionary: assigning to them
       again will overwrite them:

       .. code-block:: python

         opts.compilation_poplar_options = {"option1": "true"}
         ...
         opts.compilation_poplar_options = {"option2": "5"}

       would mean that Poplar compilation is given only ``{"option2": "5"}``.
       To achieve behaviour like the old configuration API, use the following:

       .. code-block:: python

         opts.compilation_poplar_options = {"option1": "true"}
         ...
         opts.compilation_poplar_options = {**{"option2", "5"},
                                            **opts.compilation_poplar_options}

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

Environment variable changes
''''''''''''''''''''''''''''

The ``dump_text_reports_to_stdio`` flag passed to ``TF_POPLAR_OPTIONS`` has been
deprecated and has no effect. Use the PopVision Graph Analyser to manually
inspect profiles.

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

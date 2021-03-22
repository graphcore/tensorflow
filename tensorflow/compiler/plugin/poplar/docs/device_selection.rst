Targeting the Poplar XLA device
-------------------------------

The Poplar XLA devices are named ``/device:IPU:X``, where X is an integer which
identifies that logical device. This can consist of one or more physical IPU
devices, as described below.

A Python context handler is available for setting up all appropriate scoping
while creating the graph:

.. literalinclude:: tutorial_sharding.py
  :language: python
  :linenos:
  :start-at: # Create the IPU section of the graph
  :end-at: result = ipu.ipu_compiler.compile

For very simple graphs, it is sufficient to use the IPU scope to define the
parts of the graph which will be compiled.  For most graphs, the function
``ipu_compiler.compile()`` must be used.  This must be placed inside an IPU
device scope.

The function ``ipu_compiler.compile()`` will cause all operations created by
the Python function passed into its first argument to be placed on the IPU
system, and be compiled together into a single Poplar executable.

Supported types
~~~~~~~~~~~~~~~

Poplar and the PopLibs libraries support the following data types:

*  ``tf.float32``
*  ``tf.float16``
*  ``tf.int32``
*  ``tf.bool``

Device selection
~~~~~~~~~~~~~~~~

Hardware configuration options enable you to select the number of IPU devices.
By default, TensorFlow will create one device.  This device
will be for a single IPU. The first available single IPU will be used.

Two API calls are available for selecting the number and configuration
of the IPU system:

* ``auto_select_ipus`` allows the selection of a number of IPUs. The function
  returns a single logical device containing the requested number of IPUs.

* ``select_ipus`` allows the selection of a specific IPU hardware devices using
  ID numbers as returned by the ``gc-info`` tool.

Both of these functions takes the options structure
returned by the ``create_ipu_config`` function as the first argument .

The second argument to ``auto_select_ipus`` is the number of IPUs required.

The second argument to ``select_ipus`` is either an integer or a list.

When a single integer is specified, this will be treated as the ID of the IPU
device or devices to use. The ID specifies a single IPU, if it is in the range 0 to
15. Larger numbers represent "multi-IPU" IDs that specify groups of closely
connected IPUs.

For example, to use all the IPUs in a 16-IPU system the
appropriate ID is 30. (See the `IPU Command Line Tools
<https://docs.graphcore.ai/projects/command-line-tools/>`_ document for details
of how device IDs map to available IPUs.) This will allocate a single
TensorFlow device (``/device:IPU:0``) configured with all 16 IPUs.

You can also use a list of IDs as the argument to ``select_ipus``. This
configures a TensorFlow device for each ID in the list (``/device:IPU:0``,
``/device:IPU:1``, and so on). Again, each ID value can specify a single IPU or
multiple IPUs.

For more examples, see the documentation in :ref:`api-section`.

Once the hardware structure has been specified, the API call
``ipu.utils.configure_ipu_system`` must be used to attach to and initialise the
hardware.

.. literalinclude:: tutorial_sharding.py
  :linenos:
  :language: python
  :start-at: # Configure the IPU system
  :end-at: ipu.utils.configure_ipu_system



Configuring compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``create_ipu_config`` function has many options for system configuration.
They are divided into roughly three categories:

1) Profiling and report generation.
2) IO control.
3) Graph creation.

In addition to ``auto_select_ipus`` and ``select_ipus``, several other
functions exist for configuring the hardware and compiler.

* ``set_compilation_options`` sets general options to be passed to the Poplar
  compiler.
* ``set_convolution_options``, ``set_matmul_options`` and
  ``set_pooling_options`` pass specific options directly to the PopLibs
  convolution and pooling operations.
* ``set_report_options`` passes options directly to the Poplar
  summary report generator.
* ``set_ipu_model_options`` controls the Poplar IPU Model device type.
* ``set_recomputation_options`` turns on recomputation, to reduce the memory
  requirement at the expense of speed.
* ``set_floating_point_behaviour_options`` controls the IPUs floating
  point control register.
* ``set_optimization_options`` controls the performance and memory use
  trade offs.

More options are available on the ``create_ipu_config`` function itself. These
mostly control specific features of the Poplar and PopLibs operations.
Some of the main ones are described below:

* ``scheduler_selection`` specifies the scheduling algorithm the Poplar XLA
  backend uses to schedule the instructions in the graph during the compilation
  stage.

  The available algorithms are:
  * ``CHOOSE_BEST`` (default), which compares several of the scheduling
    algorithms below and selects the one that leads to the lowest predicted
    overall peak liveness. This can sometimes produce incorrect results because
    the overall peak liveness isn't always a good measure for the maximum
    liveness on one tile of the processor.
  * ``CLUSTERING``, which groups clusters of operations together in order to
    look through stretches of instructions with potentially high liveness.
  * ``POST_ORDER``, which schedules the instructions in the order which is
    obtained by walking the graph in 'post order'.
  * ``LOOK_AHEAD``, which looks ahead a number of operations from any
    schedulable one, as given by the ``max_scheduler_lookahead_depth`` and
    ``max_scheduler_search_space_size`` options. It attempts to look through
    areas of high liveness.
  * ``SHORTEST_PATH``, which gives priority to the shortest path to the root.

* ``max_scheduler_lookahead_depth`` controls how far the ``LOOK_AHEAD``
  scheduling algorithm can look beyond a given scheduling decision to understand
  the max-liveness implications. This search space grows very quickly and can
  take an unacceptable amount of time for large values.

* ``max_scheduler_search_space_size`` introduces an upper-limit to the size of
  the ``LOOK_AHEAD`` scheduling algorithm's search space to guarantee that it
  will terminate in a reasonable amount of time.

See the documentation in :ref:`api-section` for more details.

.. _env-var-section:

TF_POPLAR_FLAGS environment variable
....................................

The options passed through ``create_ipu_config`` and ``configure_ipu_system``
can be directed at any machine in a TensorFlow cluster.  Some configuration
options are provided by an environment variable called ``TF_POPLAR_FLAGS``.

If you set ``TF_POPLAR_FLAGS=--help`` and execute a TF session, it will output some
help for each option. Some of the more common options are described below.
For a full list, refer to  :ref:`api-section`.

.. list-table:: TensorFlow configuration options
  :width: 100%
  :widths: 45,55
  :header-rows: 1
  :class: longtable

  * - Option
    - Description
  * - ``--dump_schedule_as_dot``
    - Dump the schedule of the XLA graph to the user console.
  * - ``--dump_text_reports_to_stdio``
    - If profiling is enabled, then a text summary of the profile will be dumped
      to standard output, in addition to the normal report processing.
  * - :samp:`--executable_cache_path={path}`
    - Enables the Poplar executable cache.
      See :ref:`caching_executables`.
  * - ``--fallback_scheduler``
    - Uses the standard TensorFlow scheduler, instead of the Graphcore specific
      one.
  * - ``--help``
    - Print information for all the options.
  * - :samp:`--log_cycle_count={int}`
    - Log the number of cycles used in evaluating the main graph. The numeric
      argument indicates the tile on which the cycle count operation will be
      created. This may be used as an alternative to profiling for graphs with
      dynamic control flow.
  * - :samp:`--max_compilation_threads={int}`
    - Sets the maximum number of threads which Poplar is allowed to use for
      compiling the executable.
  * - :samp:`--max_infeed_threads={int}`
    - Sets the maximum number of threads which each infeed queue is allowed to
      use when accessing data from datasets.
  * - ``--null_data_feed``
    - Cause any infeed queues to copy garbage data to the IPU rather than real
      data. This option can be used to determine whether the dataset provided to
      the infeed queue is the bottleneck during execution.
  * - :samp:`--save_interval_report={path}`
    - Dumps the Poplar interval report to the given directory.
  * - :samp:`--save_vertex_graph={path}`
    - Dumps the Poplar vertex graph (as a DOT file) to the given directory.
  * - ``--synthetic_data_initializer``
    - Used in combination with the
      ``--use_synthetic_data`` or ``--synthetic_data_categories`` option to
      control how the inputs to the graph will be initialised on the IPU.
      The values will be either random: ``--synthetic_data_initializer=random``

      Or a constant value *X*: :samp:`--synthetic_data_initializer={X}`
  * - :samp:`--tensor_map_file_path={path}`
    - Cause a JSON file containing the tile mapping of all tensors to be written
      to this directory.
  * - ``--use_ipu_model``
    - Use the Poplar IPUModel for graph compilation and execution.
  * - ``--synthetic_data_categories``
    - Prevents the system from transferring data of the given types to/from the IPU
      when executing code. This is used for testing performance without the overhead
      of data transfer.

      The values can be any of: infeed, outfeed, seed, hostembedding or parameters.

      For example, ``--synthetic_data_categories='infeed,outfeed'`` will use synthetic data just
      for in and outfeeds.
  * - ``--use_synthetic_data``
    - Prevent the system from downloading or uploading data to the IPU when
      executing code. This is used for testing performance without the overhead
      of data transfer. When enabled implies that all ``--synthetic_data_categories``
      are set.

      Executing the ``dequeue`` op for an ``IPUOutfeedQueue``
      with ``outfeed_mode`` set to ``IPUOutfeedMode.LAST`` will throw an
      exception when this flag is set.
  * - :samp:`--while_loop_brute_force_max_trip_count={int}`
    - Sets the upper bound for how many iterations a while loop will be
      simulated for in order to brute force the number of times it will be
      executed.
  * - :samp:`--show_progress_bar={true}|{false}|{auto}`
    - Whether to show the compilation progress bar. Either ``true``, ``false``
      or ``auto``. When set to ``auto``, the progress bar will only be enabled
      when attached to a console, ``VLOG`` logging is disabled and compiling a
      graph which can take more than few seconds to compile. Defaults to
      ``auto``.
  * - :samp:`--on_demand_device_poll_time={int}`
    - When using 'ON_DEMAND' connection type, configure how often to poll for
      the device (in milliseconds) when a device is not available - defaults to
      1000ms. Minimum is 100ms.
  * - :samp:`--on_demand_device_timeout={int}`
    - When using 'ON_DEMAND' connection type, configure how long to wait (in
      milliseconds) for a device before timing out - defaults to 3600000ms
      (1 hour).
  * - :samp:`--sync_replica_start`
    - Add a global synchronisation point at the start of each replica's main
      Poplar program. This can be used to force each replica to not execute
      until all replicas have started.

Multiple options can be specified at the same time by concatenating them like command line
switches, for example: ``TF_POPLAR_FLAGS=--executable_cache_path=/tmp/cache --log_cycle_count=123``.

Supported operations
~~~~~~~~~~~~~~~~~~~~

A list of supported TensorFlow operations is provided in :ref:`supported-section`.

Unsupported operations
~~~~~~~~~~~~~~~~~~~~~~

TensorFlow core operations which use variable buffers or strings are not
supported. For instance, ``JpegDecode``.

Unsupported operations will cause the compilation to fail.

By including
``config=tf.ConfigProto(log_device_placement=True)`` as an argument to the
creation of the session, you can check whether the operations in your graph
have been targeted at the Poplar device. For example:

.. code-block:: python

  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

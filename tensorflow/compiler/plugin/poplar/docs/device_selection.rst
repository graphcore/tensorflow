Targeting the Poplar XLA device
-------------------------------

The Poplar XLA devices are named ``/device:IPU:X``, where X is an integer which
identifies that logical device. This can consist of one or more physical IPU
devices, as described below.

A Python context handler is available for setting up all appropriate scoping
while creating the graph:

.. literalinclude:: tutorial_sharding.py
  :language: python
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

Poplar and the Poplibs libraries support the following data types:

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
<https://documents.graphcore.ai/documents/UG10/latest>`_ document for details
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
  ``set_pooling_options`` pass specific options directly to the Poplibs
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
mostly control specific features of the Poplar and Poplibs operations.
Some of the main ones are described below:

* ``max_scheduler_lookahead_depth`` controls how far the scheduler can look
  beyond a given scheduling decision to understand the max-liveness
  implications. This search space grows very quickly and can take an
  unacceptable amount of time for large values.
* ``max_scheduler_search_space_size`` introduces an upper-limit to the size of
  the schedule search space to guarantee that it will terminate in a reasonable
  amount of time.

* ``scheduler_selection`` controls the particular scheduler that is selected
  to perform the scheduling of instructions in the compilation stage.  By
  default, several schedules will be created and the one with the lowest
  predicted liveness chosen.  This can sometimes produce incorrect results
  because the overall peak liveness isn't always a good measure for the maximum
  liveness on one tile of the processor.

  The available schedulers are:

  * ``Clustering``, which groups clusters of operations together in order to
    look through stretches of instructions with potentially high liveness.
  * ``PostOrder``, which schedules the instructions in the order which is
    obtained by walking the graph in 'post order'.
  * ``LookAhead``, which looks ahead a number of operations from any
    schedulable one, as given by the ``max_scheduler_lookahead_depth`` and
    ``max_scheduler_search_space_size`` options described above.  It attempts
    to look through areas of high liveness.
  * ``ShortestPath``, which schedules the graph giving priority to
    the shortest path to the root.

See the documentation in :ref:`api-section` for more details.

.. _env-var-section:

TF_POPLAR_FLAGS environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The options passed through ``create_ipu_config`` and ``configure_ipu_system``
can be directed at any machine in a TensorFlow cluster.  Some configuration
options are provided by an environment variable called ``TF_POPLAR_FLAGS``.

If you set ``TF_POPLAR_FLAGS=--help`` and execute a TF session, it will output some
help for each option. Some of the more common options are described below.
For a full list, refer to  :ref:`api-section`.

* ``--help`` will print the information for all the flags.

* ``--use_synthetic_data`` will prevent the system from downloading or uploading
  data to the card when executing code.  This is used for testing performance
  without the overhead of data transfer.

* ``--synthetic_data_initializer`` is used in combination with the
  ``--use_synthetic_data`` flag to control how the inputs to the graph will be initialised
  on the IPU. The values will be either random (``--synthetic_data_initializer=random``)
  or a constant value ``X`` (``--synthetic_data_initializer=X``)

* ``--use_ipu_model`` will use the Poplar IPUModel for graph compilation and
  execution.

* ``--log_cycle_count`` will log the number of cycles used in evaluating the
  main graph. The numeric argument indicates the tile on which the cycle count
  operation will be created.
  This may be used as an alternative to profiling
  for graphs with dynamic control flow.

* ``--while_loop_brute_force_max_trip_count`` is the upper bound for how many
  iterations a while loop will be simulated for in order to brute force the
  number of times it will be executed.

* ``--max_compilation_threads`` sets the maximum number of threads which Poplar
  is allowed to use for compiling the executable.

* ``--max_infeed_threads`` sets the maximum number of threads which each infeed
  queue is allowed to use when accessing data from datasets.

* ``--save_vertex_graph`` dumps the Poplar vertex graph (as a DOT file) to the given
  directory.

* ``--save_interval_report`` dumps the Poplar interval report to the given
  directory.

* ``--executable_cache_path`` enables the Poplar executable cache.
  See :ref:`caching_executables`.

* ``--save_interval_report`` dumps the Poplar interval report to the given
  directory.

* ``--tensor_map_file_path`` will cause a JSON file containing the tile mapping
  of all tensors to be written to this directory.

* ``--dump_schedule_as_dot`` will dump the schedule of the XLA graph to the user
  console.

* ``--fallback_scheduler`` uses the standard TensorFlow scheduler, instead of
  the Graphcore specific one.

* ``--allow_nans`` will allow NaNs.

* ``--null_data_feed`` will cause any infeed queues to copy garbage data to the
  IPU rather than real data. This option can be used to determine whether the
  dataset provided to the infeed queue is the bottleneck during execution.

* ``--dump_text_reports_to_stdio`` if profiling is enabled, then a text summary
  of the profile will be dumped into the standard output, in addition to the
  normal report processing.

Multiple options can be specified at the same time by concatenating them like command line
switches, for example: ``--executable_cache_path=/tmp/cache --allow_nans``.

.. _caching_executables:

Caching of compiled executables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It can take a long time to compile a large fused graph into an executable
suitable for the IPU.  To prevent the need for compiling every time a
TensorFlow process is started, you can enable an executable cache.

You can use the flag ``--executable_cache_path`` to specify a directory where
compiled files will be placed.  Fused XLA/HLO graphs are hashed with a 64-bit
hash and stored in this directory. For example:

.. code-block:: python

  TF_POPLAR_FLAGS='--executable_cache_path=/tmp/cachedir'

A pair of files will be saved for each compiled graph, the TensorFlow
metadata and the Poplar executable.

The cache does not manage the files within the directory. It is your
responsibility to delete files.  No index is kept of the
files, so they can be deleted without risk.

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

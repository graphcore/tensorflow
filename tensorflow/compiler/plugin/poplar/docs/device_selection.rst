.. _device_selection:

Targeting the Poplar XLA device
-------------------------------

The Poplar XLA devices are named ``/device:IPU:X``, where X is an integer which
identifies that logical device. This can consist of one or more physical IPU
devices, as described below.

A Python context handler is available for setting up all appropriate scoping
when you create the graph. This will place all operations built inside it on the
chosen Poplar XLA device:

.. code-block:: python
  :linenos:

  with ipu_scope("/device:IPU:0"):
    xla_result = ipu.ipu_compiler.compile(my_net, [x_data, y_data, p_angle])

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
By default, TensorFlow will create one virtual device (``/device:IPU:0``) with
a single IPU. The first available single IPU will be used.

Two API options on the :py:class:`~tensorflow.python.ipu.config.IPUConfig` are
available for controlling which or how many IPUs this virtual device will use:

* ``auto_select_ipus`` allows you to specify a quantity of
  IPUs to use. The virtual device will be given that many IPUs.

* ``select_ipus`` allows you to choose a specific IPU hardware
  device using its ID. The device IDs can be seen with the ``gc-info`` command
  line tool. An ID can represent a single IPU device or a larger "multi-IPU"
  device that contains a group of closely connected single IPU devices.

  For example, the largest single IPU device in a 16-IPU system has the ID 15,
  while the largest multi-IPU device has the ID 30.
  (See the `IPU Command Line Tools
  <https://docs.graphcore.ai/projects/command-line-tools/>`_ document for
  details of how device IDs map to available IPUs.)

You can also pass a list (or tuple) to either of these options. This will
configure a separate TensorFlow virtual device for each value in the list
(``/device:IPU:0``, ``/device:IPU:1``, and so on).

Once the hardware structure has been specified, the
:py:meth:`~tensorflow.python.ipu.config.IPUConfig.configure_ipu_system` method
of the config object must be used to attach to and initialise the hardware:

.. code-block:: python
  :linenos:

  from tensorflow.python import ipu

  cfg = ipu.config.IPUConfig()
  # Select multi-IPU with device ID 30, which contains all 16 IPUs of a 16-IPU system.
  cfg.select_ipus = 30
  cfg.configure_ipu_system()

This example will allocate a single TensorFlow virtual device
(``/device:IPU:0``) which will use all 16 IPUs in a 16-IPU system.

For more examples, see the documentation for the options in the
Python API: :ref:`auto_select_ipus <auto_select_ipus>`, :ref:`select_ipus <select_ipus>`.

.. _configuring-section:

Configuring system options
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to ``auto_select_ipus`` and ``select_ipus``, the
:py:class:`~tensorflow.python.ipu.config.IPUConfig` class has many other
options for system configuration. The class is a nested structure of attributes
which organises these options into several categories and sub-categories. To use
it, instantiate it and treat its attributes like ordinary Python variables:

.. code-block:: python
  :linenos:

  # Initialize an IPUConfig instance
  cfg = ipu.config.IPUConfig()

  # Ask for 2 IPUs on /device:IPU:0
  cfg.auto_select_ipus = 2

  # Change our mind and decide we need 4 IPUs instead. This is fine since
  # setting any config attribute has no effect until the config is used to
  # configure the IPU system
  cfg.auto_select_ipus = 4

  # Configure the system with the config, creating /device:IPU:0 with 4 IPUs
  cfg.configure_ipu_system()

Some attributes are not configuration options themselves, but rather names for
general categories of grouped options. Categories cannot be set.
You can access an arbitrarily nested attribute with chained dot notation, and
an attribute's full name indicates exactly where it is in the
:py:class:`~tensorflow.python.ipu.config.IPUConfig` nested structure.
For example:

.. code-block:: python
  :linenos:

  cfg = ipu.config.IPUConfig()

  # Set the IPU Model version, which is in the "ipu_model" category
  # Its full name is "ipu_model.version"
  cfg.ipu_model.version = "ipu2"
  print(cfg.ipu_model.version)  # ipu2

  # Set the multi replica distribution process count, which is in the
  # "multi_replica_distribution" sub-category of the "experimental" category
  # of the config
  cfg.experimental.multi_replica_distribution.process_count = 2
  print(cfg.experimental.multi_replica_distribution.process_count)  # 2

  # You cannot set to a category, since it's a grouping of options and is not an
  # option itself
  cfg.experimental = 2  # Will error

  cfg.configure_ipu_system()

Options are type checked when they're set and if an option cannot be
found, then a similarly spelled one may be suggested. Additionally, setting to a
deprecated option will give you a warning:

.. code-block:: python
  :linenos:

  cfg = ipu.config.IPUConfig()

  # Try to set an option to an incorrect type
  cfg.ipu_model.version = True  # Will error immediately asking for a string
  # Make a spelling mistake when writing the option name
  cfg.ipu_model.vrsion = "ipu2"  # Will ask if you meant "version"

The metadata for any attribute, including its default, allowed types,
docstring, full name and whether or not it is deprecated can all be accessed
through the
:py:meth:`~tensorflow.python.ipu.config.IPUConfig.get_attribute_metadata`
function, which takes a string representing the attribute's full name, relative
to the category you are calling the function on. For example:

.. code-block:: python
  :linenos:

  cfg = ipu.config.IPUConfig()

  # Access by full name from the base config:
  metadata = cfg.get_attribute_metadata("ipu_model.version")
  # Access by name relative to the "ipu_model" sub-category:
  metadata = cfg.ipu_model.get_attribute_metadata("version")

  # Use the metadata
  print(metadata.types)  # [str]
  print(metadata.default)  # "ipu2"
  print(metadata.deprecated)  # False indicates it is not deprecated
  print(metadata.__doc__)  # "Specify the ipu version to be used by the..."

  # Check a value against the option's type
  metadata.check_type(5)  # Will fail, since this option needs a string.

  # Print a deprecation message if the option is deprecated
  metadata.warn_if_deprecated()

This is useful for forwarding IPU configuration options to command line
interfaces in applications. Note that you can also access the metadata of
categories themselves, but the ``types`` and ``default`` fields will be empty.
You can see a full description of the available metadata in the
:py:class:`~tensorflow.python.ipu.config.AttributeMetadata` class.

When you are finished adjusting the
:py:class:`~tensorflow.python.ipu.config.IPUConfig` instance, use it to
configure the IPU system by calling its
:py:meth:`~tensorflow.python.ipu.config.IPUConfig.configure_ipu_system` method.
The options set on an instance will not have any effect until this is done. Note
that configuring the system does not alter the instance.
For example:

.. code-block:: python
  :linenos:

  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = 4
  cfg.ipu_model.compile_ipu_code = False
  cfg.ipu_model.version = "ipu2"
  cfg.scheduling.algorithm = ipu.config.SchedulingAlgorithm.Clustering
  ...

  cfg.configure_ipu_system()
  # The IPU system can now be used.

  # The config can still be accessed after configuration.
  print(cfg.ipu_model.version)  # ipu2

In addition to ``auto_select_ipus`` and ``select_ipus``, some other options on
the :py:class:`~tensorflow.python.ipu.config.IPUConfig` which can be used to
configure the hardware and compiler are highlighted below:

* :ref:`allow_recompute <allow_recompute>` turns on recomputation, to reduce the memory requirement
  of the model at the expense of speed.
* :ref:`selection_order <selection_order>` to control the mapping between the "virtual" IPUs and
  physical IPUs of a multi-IPU device.
* :ref:`compilation_poplar_options <compilation_poplar_options>` sets general options to be passed to the Poplar
  compiler.
* :ref:`convolutions.poplar_options <convolutions.poplar_options>`, :ref:`matmuls.poplar_options <matmuls.poplar_options>` and
  :ref:`pooling.poplar_options <pooling.poplar_options>` pass specific options directly to the PopLibs
  convolution, matmul and pooling operations.
* :ref:`ipu_model <ipu_model>` is a category containing options that control the Poplar IPU
  Model device type.
* :ref:`floating_point_behaviour <floating_point_behaviour>` is a category containing options that allow you
  to configure the IPU device's floating point control register.
* :ref:`optimizations <optimizations>` is a category containing options that can toggle various
  optimizations, which generally have a performance or memory use trade-off.
* :ref:`scheduling <scheduling>` contains options that specify and control the scheduling
  algorithm the Poplar XLA backend uses to schedule the operations in the graph
  before it is lowered to Poplar.

To view the full list, see
:py:class:`~tensorflow.python.ipu.config.IPUConfig`.

.. _env-var-section:

TF_POPLAR_FLAGS environment variable
....................................

The options passed through the ``IPUConfig`` are tied to the application that
uses that config to configure the IPU system.  Some configuration options are
instead provided by an environment variable called ``TF_POPLAR_FLAGS``.

If you set ``TF_POPLAR_FLAGS=--help`` and execute a TF session, it will output
some help for each option. The available options are described below:

.. list-table:: TensorFlow configuration options
  :width: 100%
  :widths: 45,55
  :header-rows: 1
  :class: longtable

  * - Option
    - Description
  * - ``--dump_schedule_as_dot``
    - Dump the schedule of the XLA graph to the user console.
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
  * - :samp:`--tensor_map_file_path={path}`
    - Cause a JSON file containing the tile mapping of all tensors to be written
      to this directory.
  * - ``--use_ipu_model``
    - Use the Poplar IPUModel for graph compilation and execution.
  * - ``--use_synthetic_data``
    - Prevent the system from downloading or uploading data to the IPU when
      executing code. This can be useful for testing performance without the
      overhead of data transfer.

      Using this option, all data transfer is prevented. You can instead use
      ``--synthetic_data_categories`` to prevent the transfer of
      specific categories of tensor data.

      When using this option, the graph's transferred input tensors will never
      be initialized and can therefore have undefined content. You can avoid
      this with the ``--synthetic-data-initializer`` option.

      The outputs from any outfeeds will also be uninitialized tensors on the
      host which may also contain undefined content.

      This option cannot be used when dequeuing an ``IPUOutfeedQueue`` which is
      in ``IPUOutfeedMode.LAST`` mode.
  * - ``--synthetic_data_categories``
    - Prevent the system from downloading or uploading data of the given types
      to the IPU when executing code. This can be useful for testing performance
      without the overhead of data transfer.

      The values can be any of: infeed, outfeed, seed, hostembedding or
      parameters.

      For example, ``--synthetic_data_categories='infeed,outfeed'`` will use
      synthetic data just for infeeds and outfeeds.

      When using this option, the graph's transferred input tensors will never
      be initialized and can therefore have undefined content. You can avoid
      this with the ``--synthetic-data-initializer`` option.

      This option is a more selective alternative to ``--use_synthetic_data``;
      you shouldn't specify both.
  * - :samp:`--synthetic_data_initializer={X}`
    - When using synthetic data, by default, the graph's input tensors will not
      be initialized and therefore will have undefined content.
      You can use this option to initialize these tensors on the device.

      The argument ``X`` can be set to ``uniform``, ``normal``, or a number.

      When ``uniform`` is specified, each input tensor is initialized with
      uniformly distributed random numbers (of the numerical type of the tensor).
      The range of the uniform distribution is between the
      minimum and maximum representable numbers for the specific numerical
      type of each tensor (for example, for FP16, the range would be [-65504.0,
      +65504.0], whereas for uint16, it would be [0,65535]).

      When ``normal`` is specified, each input tensor is initialized with random
      numbers drawn from the Gaussian distribution of mean 0 and
      standard deviation 1, when the tensor type is floating point.
      For integral types, a constant of value 1 is used instead.

      Finally, when the argument ``X`` is a number, only its integer part is
      used to initialize the tensors.

      For the ``--synthetic_data_initializer`` option to have an effect, you
      must also specify ``--use_synthetic_data`` **or**
      ``--synthetic_data_categories``.
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
  * - :samp:`--ipu_model_tiles={int}`
    - When specified and when using the Poplar IPUModel target, sets the number
      of tiles for the IPUModel device created. This flag has no effect if the
      ``--use_ipu_model`` flag is not used. This flag is ignored if the
      ``IPUConfig.ipu_model.tiles_per_ipu`` is set.
  * - :samp:`--sync_replica_start`
    - Add a global synchronisation point at the start of each replica's main
      Poplar program. This can be used to force each replica to not execute
      until all replicas have started.
  * - :samp:`--disable_poplar_version_check`
    - If set, the Poplar version check will be disabled.

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

.. _xla_runtime_error_handling:

Error Handling
~~~~~~~~~~~~~~

.. note::

  This section only applies to the execution using the XLA/Poplar runtime. If
  you are using the IPU embedded application runtime see
  :ref:`ea_runtime_error_handling`.

The error and exception handling by TensorFlow is divided into two categories:

* Poplar graph construction and compilation errors which occur during
  construction and compilation of TensorFlow programs.
* Poplar runtime errors which occur during the execution of the compiled
  program.

The following sections describe the actions you need to take when these errors
occur.

Construction and compilation errors
...................................

These errors are reported to the user using the TensorFlow Status error classes.
The error messages contain information about why the error occurred and what
action the user is required to take in order to stop the error from occurring.

Runtime errors
..............

These errors and exceptions occur when running a Poplar program. The full list
of all the exceptions and their meanings can be found in the Poplar
documentation in the `Exceptions <https://docs.graphcore.ai/projects/poplar-api/page/poplar/execution/exceptions.html>`__
section of the Poplar API reference manual.

These runtime errors are handled in the following manner:

* ``application_runtime_error`` - a ``tensorflow.errors.InternalError`` error
  is raised. The error message contains the reason why the error occurred. An
  IPU reset will be performed before the next execution of a Poplar program.
* ``recoverable_runtime_error`` with a recovery action ``poplar::RecoveryAction::IPU_RESET`` - a ``tensorflow.errors.InternalError`` error
  is raised. The error message contains the reason why the error occurred. An
  IPU reset will be performed before the next execution of a Poplar program.
* Unknown runtime errors - a ``tensorflow.errors.Unknown``  error
  is raised. The error message might contain the reason why the error occurred.
  When these errors occur manual intervention is required before the system is
  operational again.
* All other runtime errors - a ``tensorflow.errors.InternalError`` error
  is raised. The error message might contain the reason why the error occurred.
  When these errors occur manual intervention might be required before the
  system is operational again. The error message might contain a required
  recovery action.

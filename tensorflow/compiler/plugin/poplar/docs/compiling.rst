Compiling and pre-compiling executables
---------------------------------------

.. _caching_executables:

Caching of compiled executables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It can take a long time to compile a large TensorFlow graph into an executable
suitable for the IPU. To prevent the need for compiling the same graphs every
time a TensorFlow process is started, you can enable an executable cache.

To enable it, you can use the option ``--executable_cache_path`` to specify a
directory where the compiled executables for TensorFlow graphs will be placed.
For example:

.. code-block:: bash

  TF_POPLAR_FLAGS='--executable_cache_path=/tmp/cachedir'

An executable binary file with a file extension ``.poplar_exec`` will be saved
for each XLA/Poplar graph required to execute a TensorFlow graph.

The cache does not manage the files within the directory. It is your
responsibility to delete files. No index is kept of the files, so they can be
deleted without risk.

.. _precompiling_executables:

Pre-compiling executables
~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using a machine which is not attached to any IPU devices, but would
still like to pre-compile your TensorFlow graphs, you can do so by enabling the
pre-compile mode. In this mode your TensorFlow program is traced as if it was
executing on IPU device(s) to identify which programs need to be compiled along
with which ``tf.Variables`` are used.

During the tracing in the pre-compile mode your TensorFlow program is executed
as if it was attached to IPU device(s), however any numerical results returned
are set to zero. This means that if any operations in your TensorFlow program
are executed conditionally dependent on the previous output, they might not be
pre-compiled.

To enable the pre-compile mode, you need to use the option
``--executable_cache_path`` to specify a directory where the compiled
executables for TensorFlow graphs will be placed.
For example:

.. code-block:: bash

  TF_POPLAR_FLAGS='--executable_cache_path=/tmp/executables'

Then in your TensorFlow program you need to modify your IPU system configuration
to use the pre-compile mode.
For example:

.. literalinclude:: pre_compile_example.py
  :language: python
  :linenos:
  :emphasize-lines: 12-17

In the above example we create an IPU system configuration with pre-compile mode
for a single IPU device (IPU version 2) and with remote buffers enabled, with
the rest of the program unchanged.

.. note::
  It is important to check whether your target system supports remote buffers as
  this is required for features such as optimizer state offloading. To check,
  run the command:

  .. code-block:: console

    $ gc-info -d 0 -i

  If you see ``remote buffers supported: 1`` in the output, that means that remote
  buffers are supported on your system. For more information, see the
  `gc-info documentation <https://docs.graphcore.ai/projects/command-line-tools/en/latest/gc-info_main.html>`__.

During the execution of the program, messages will appear with the information
about what executables have been compiled and where they have been saved to.
For example:

.. code-block:: bash

  A pre-compiled Poplar program has been saved to /tmp/executables/277a08fe4c20b50.poplar_exec

Once your program has finished executing, you can copy all the executables to a
machine with IPUs.
After these have been copied, on the machine with IPUs, you should set
``--executable_cache_path`` to the directory where the compiled executables for
your TensorFlow program were copied to and then run your TensorFlow program
(without enabling the pre-compile mode).

Unsupported Operations
######################

TensorFlow programs which contain the following cannot be
pre-compiled:

* Custom user operations for which ``is_hashable`` has not been set to ``True``
  (see :ref:`custom_op_metadata`).

* Programs containing ``tensorflow.python.ipu.scopes.outside_compilation_scope``.

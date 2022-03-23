Retrieving information about compilation and execution
------------------------------------------------------

When developing models for the IPU, it is important to be able to see how
compute tiles are being used and what the balance of memory use across
them is. In certain cases, such as when investigating memory over-consumption of a
model or investigating any tile imbalance issues, it is useful to produce a
trace report that will show a number of different aspects of graph
deployment on the IPU.

To retrieve trace information about the Poplar IPU compilation and execution,
there are environment variables provided by Poplar itself to dump the
compilation and execution reports into a file. See the :ref:`Capturing IPU Reports <report_capture>`
chapter in the PopVision User Guide for more information. To enable
time-based profiling of events, see the :ref:`Capturing Execution Information <{HelpTopic.CapturingData}>`
chapter in the PopVision User Guide for more information.


TensorFlow options for reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some tracing and reporting options are provided by TensorFlow as standard, and
can be useful when developing graphs for the IPU.

``TF_CPP_MIN_VLOG_LEVEL`` is an environment variable that enables the logging of
the main C++ backend.  Setting ``TF_CPP_MIN_VLOG_LEVEL=1`` will show a lot of
output.  Included in this is the compilation and execution of the IPU code.
The output of ``TF_CPP_MIN_VLOG_LEVEL`` can be overwhelming. If only the Poplar
backend specific files are of interest, setting ``TF_POPLAR_VLOG_LEVEL=1`` will
filter the logging such that only those files produce outputs. Note that
increasing the ``VLOG_LEVEL`` of either of those environment variables will
increase the verbosity of the logs.

``TF_CPP_VMODULE`` provides a mechanism to reduce the logging to certain
translation units (source files).  This combination is quite useful:

.. code-block:: python

  TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1'

Finally, there is an environment variable called ``XLA_FLAGS`` which provides
options to the general XLA backend. For example, the follow will produce a
Graphviz DOT file of the optimised HLO
graph which is passed to the Poplar compiler.

.. code-block:: python

  XLA_FLAGS='--xla_dump_to=. --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re=forward-allocation --xla_hlo_graph_sharding_color'

The HLO pass ``forward-allocation`` is one of the final passes to run before the HLO
instructions are scheduled for passing to the Poplar graph compiler.
Running with these options will create a file
called something like
``module_0001.0001.IPU.after_forward-allocation.before_hlo-memory-scheduler.dot``.
(The way that the file names are generated is explained in :ref:`xla_file_naming`.)
The Graphviz ``dot`` command can be used to convert this data to an image.

More information on the XLA flags can be found in the definition of the XLA proto here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto

.. _xla_file_naming:

XLA graph file naming
~~~~~~~~~~~~~~~~~~~~~

The number of files produced depends on the number of TensorFlow HLO modules
generated. This can generally be predicted from the number of ``sess.run`` calls
on distinct graphs that you make. For example, if your program contains a variable
initialisation then this will be compiled as a separate XLA graph
and appear as a separate file when dumped. If your program creates a report operation,
then that will also be compiled as a separate XLA graph.

When you use ``ipu_compiler.compile``, you force everything inside the compile
call to be compiled into a single XLA graph. If you don't use
``ipu_compiler.compile``, then the results depend on the XLA scheduler, which
will combine or split up parts of the TensorFlow graph as it sees fit, creating
many arbitrary distinct XLA graphs. If you do not use ``ipu_compiler.compile``,
expect to see a larger number of XLA graphs generated. Please note, there is no guarantee your
compiled op will only produce one XLA graph. Sometimes others are created for
operations such as casting.

The following description provides a break down of the names of the generated files.
These are of the general form:

  ``module_XXXX.YYYY.IPU.after_allocation-finder.before_forward-allocation.dot``

* There is always a ``module_`` prefix, which indicates that this
  is the graph for an HLO Module.

* The first ``XXXX`` is the HLO module's unique ID, generated here:
  https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/compiler/xla/service/dump.cc#L263

  There is no guarantee about the spacing between IDs, only that they are unique
  and increasing.

* To understand the rest of the name, ``YYYY.IPU.......dot``, we need to
  understand that the XLA graph is operated on by multiple different HLO passes,
  each modifying the XLA graph by optimizing, shuffling or otherwise rewriting it.
  After these passes, the graph is then lowered to Poplar. There are some
  TensorFlow native HLO passes, and there are some IPU specific ones.

  When dumping the XLA graphs, we can render the XLA graph before and after any
  HLO pass (for example, to see the effect of that pass on the graph) by
  supplying the argument ``--xla_dump_hlo_pass_re=xxxx``, where ``xxxx`` is a
  regular expression describing which passes you want. TensorFlow will then
  render the XLA graph before and after every pass whose name matches that regex.
  For example, if you wanted to see the effect of every XLA HLO IPU
  pass involving while loops, you could use ``--xla_dump_hlo_pass_re=*While*``.

  The number ``YYYY`` is simply an ID related to the order in which these graphs
  are generated.

* Finally, the passes which the graph was "between" when it was rendered
  are appended to the filename.

  The ``before_optimizations`` graph is always rendered if dumping XLA.

* The HLO modules have CamelCase class names by convention. For the file names,
  these are converted to snake_case.


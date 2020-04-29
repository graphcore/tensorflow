Retrieving information about compilation and execution
------------------------------------------------------

When developing models for the IPU, it is important to be able to see how
compute tiles are being used and what the balance of memory use across
them is. In certain cases, such as when investigating memory over-consumption of a
model or investigating any tile imbalance issues, it is useful to produce a
trace report that will show a number of different aspects of graph
deployment on the IPU.

Several mechanisms are available to retrieve trace information about the
Poplar IPU compilation and execution. Firstly, there are environment variables
provided by Poplar itself to dump the compilation and execution reports into a
file. See the "Profiling" chapter in the
`Poplar and Poplibs User Guide <https://documents.graphcore.ai/documents/UG1/latest>`_
for more information.

Within TensorFlow, the basic steps for this are:

* Include an operation in the graph to retrieve the reports
* Enable tracing in the hardware configuration options
* Execute the graph, including the operation to retrieve the reports
* Extract the reports from the returned events

Adding an operation to get compilation and execution events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two operations are available to fetch events from the Poplar backend. The first
is an operation which fetches the reporting events into a tensor, and is
typically executed independently of the main graph.  The second is a summary
event which will extract the reports along with any other summary events. These
events will typically be written into a file using the
``tensorflow.summary.FileWriter`` class.

ipu_event_trace()
_________________

This is an operation which retrieves all IPU events since the last time it was
executed. The operation must be placed on the CPU, and returns the events as a
one dimensional tensor of strings containing serialised IPU event protobufs,
from ``tensorflow.compiler.plugin.poplar.driver.trace_pb2.IpuTraceEvent``.

This is the example from the tutorial with a few lines of additional code to
create a trace report:

.. literalinclude:: logging_example_1.py
  :language: python
  :linenos:

The example starts by importing two new elements that are IPU-specific APIs.
The first import is ``gen_ipu_ops``, which will generate the event trace.
The second import is an assortment of utility functions, one of
which is used here to parse the event trace to a readable output.

The event trace operation is created when ``gen_ipu_ops`` is called to instantiate
the trace and returns it to ``report``. This is then fed to the TensorFlow session
as a ``run`` argument, directly following the session run call to the feed-forward
pass through ``basic_graph``. In essence, the report is generated based on the last
session graph call. The trace output is then parsed through
``extract_all_strings_from_event_trace``, and a log file is generated. The final
step of writing the trace to a file is done near the end of the
example where a file is opened and the parsed trace data written to it.

ipu_compile_summary(name, [op list])
____________________________________

This produces a summary which can be tied into the rest of the summary system
to produce output for Tensorboard. The parameter ``name`` is the name of the
summary, and ``op`` is one of the operations in the IPU graph. It is best to choose either
the inference output for an inference graph, the loss output for an evaluation
graph, or the train op for a training graph.

.. code-block:: python

  import tensorflow as tf
  from tensorflow.python import ipu

  ...

  tf.summary.scalar('c_out', c)
  ipu.summary_ops.ipu_compile_summary('report', [c])
  all_sum = tf.summary.merge_all()

  ...

  f = tf.summary.FileWriter('logs')
  with tf.Session() as s:
    sum_out, ... = s.run([all_sum, ...])
    f.add_summary(sum_out, 0)

    print("c = {}".format(c))


Enabling tracing in the hardware configuration options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main function for producing an IPU system hardware configuration is called
``create_ipu_config``.  It provides several options for controlling the logging
and tracing of Poplar compilations.

* ``profiling``: This enables compilation and execution graph reports in Poplar,
  and generates ``COMPILE_BEGIN`` and ``COMPILE_END`` events in the trace.

* ``enable_ipu_events``: Setting this to ``True`` while leaving ``profiling`` as
  ``False`` will generate trace events without creating the Poplar compilation
  and execution reports in them.  This is useful for getting timing information
  from the event trace without the overhead of the Poplar reporting.

* ``use_poplar_text_report``: Normally, the Poplar reports are generated in JSON
  format.  Setting this parameter to ``True`` will generate a text summary report
  instead of JSON.

* ``use_poplar_cbor_report``: Instead of a JSON format report, a CBOR format
  report will be generated.

* ``profile_execution``: When this is set to ``True``, then EXECUTE events will
  be generated in addition to compilation events.  By default the execution events
  will contain a ``device`` type trace.  If a different type of execution trace
  is required, then instead of ``True``, one of ``ExecutionProfileType.DEVICE_PROFILE``,
  ``ExecutionProfileType.IPU_PROFILE`` or ``ExecutionProfileType.TILE_PROFILE``
  can be used.

* ``enable_poplar_serialized_graph``: Setting this to ``True`` will include a
  serialized Poplar graph in the COMPILE_END event.  Typically these are very
  large, so this feature is disabled by default.  The result can be viewed
  using the PopVision Graph Analyser.

* ``report_every_nth_execution``: This will restrict the number of execution
  reports to a subset of all executions.

* ``max_report_size``: Poplar reports can get very large.  This parameter can
  be used to restrict the maximum size of report generated.  Reports larger
  than this value will be discarded and a warning message sent to the TensorFlow log.

* ``report_directory``: Rather than reports being placed directly into the events,
  they can be written to a file, and the file name written into the event log.
  This behaviour is enabled by setting this parameter to a directory name. The
  reports will be written into subdirectories inside the `report_directory`
  where each subdirectory is for an XLA graph compilation.


Extract the reports from the returned events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the summary event generator has been used then the events will be inside
``Tensor`` type events in the Tensorboard logs.

.. A tool for extracting these from the log, called ``extract_logs.py``, is
.. available in the Graphcore "Toolshed" repository on GitHub.

If the individual report gathering event is used then executing it will return
an array of tensors.  Within each tensor is a string which is an ``IpuTraceEvent``
of one type.

The ``IpuTraceEvent`` is within the ``tensorflow`` namespace at
``tensorflow.compiler.plugin.poplar.driver.trace_pb2.IpuTraceEvent``.  It is
a protobuf that can be decoded from the string into an object with fields
containing trace information.

Several utility functions are available for extracting fields, for example:

.. code-block:: python

  rep = sess.run(report)
  compile_reports = ipu.utils.extract_compile_reports(rep)
  execute_reports = ipu.utils.extract_execute_reports(rep)
  poplar_graphs = ipu.utils.extract_poplar_serialized_graphs(rep)
  events = ipu.utils.extract_all_events(rep)

See the :ref:`api-section` section for more information.


Producing reports for use with the PopVision Graph Analyser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure Tensorflow to produce a directory containing files named
correctly for the PopVision Graph Analyser, you should enable the following
features in the ``create_ipu_config`` options:

* Set `profiling` to True
* Set `profile_execution` to True, if you would like an execution profile.
* Set `enable_poplar_serialized_graph` to True if you would like a Poplar
  graph.
* Set `report_directory` to a suitable directory where you want profiling
  information to be placed.

The system will then create subdirectories inside the configured one, with
each subdirectory containing a single graph compilation.  The directories
are named after the automatically generated Tensorflow operation clusters.

A simple way to discover which of the subdirectories is the one containing
the main graph is to look for the one containing the largest report.json
file.

Note that while the EXECUTE trace events contain every execution profile,
if you dump execution profiles straight to disk, they will go into a file
with a fixed filename, and each execution of the model - for instance by
calling Session.run() in Tensorflow 1.x - will overwrite the file for the
previous execution.

Also be aware that the default size beyond which a report or graph file will
be discarded is 0x10000000 bytes.  For particularly large models, the file
size can exceed this, in which case you can change the maximum file size by
including the ``max_report_size`` parameter in the ``create_ipu_config``
call.


COMPILE_BEGIN
_____________

This event is generated when the Poplar compilation begins.  It contains the
XLA module name, a timestamp and the ordinal of the device that the code was
compiled for.

COMPILE_END
___________

This is generated when the Poplar compilation ends.  It contains the module
name, a timestamp, an ordinal and the following compilation trace fields:


* ``compilation_report`` is the Poplar compilation report.
* ``duration`` is the duration of the compilation.
* ``tensor_map`` is a mapping of tensors generated by XLA/HLO instructions to
  the IPU tiles where those tensors are mapped.
* ``poplar_graph`` is the serialized form of the Poplar graph.

Tensor map
..........

The ``tensor_map`` field has the following format. It is JSON but, in order to
keep it dense, it is mostly JSON lists instead of keyed dictionaries.

At the top level there is a map called ``mappings`` which contains an entry for
each XLA computation, keyed by the name of that computation.  The value is a
list of tensors generated by that computation.

.. code-block:: python

  { 'mapping' : {'computation_0' : [ ... ], 'computation_1' : [ ... ] } }

Each tensor in that list is also a list, consisting of the following items:

* 0 - name of the XLA/HLO instruction generating the tensor.
* 1 - the ordinal of the tensor produced by that instruction.
* 2 - a list of integers indicating the shape of the tensor.
* 3 - a string indicating the tensor element type.
* 4 - a Boolean indicating if the tensor contains any constant elements.
* 5 - a Boolean indicating if the tensor contains any aliases.
* 6 - the total number of elements in the tensor.
* 7 - a list of information about the elements on each tile, for example:

  .. code-block:: python

    [ 'add.0', 0, [32, 32], 'float', 0, 0, 2, 256, [ ... ] ]

The list of elements on each tile has one entry per tile that contains
elements of the tensor. Each entry is itself a list, containing the following
items:

- the tile index number.
- the total number of elements on that tile.

The ``instruction_info`` field contains information about how the specific
HLO instructions were mapped to Poplar API calls.  Its format is as follows:

.. code-block:: python

  { 'ml_types': {'instruction': <ml_type>, ... } }

The instruction is the name of the instruction at the HLO level, which
is similar to the name in the main compilation report.  The ``ml_type`` field
takes one of the following values, for instructions which are convolution or
matmul:

* 0 - Unclassified
* 1 - Standalone
* 2 - The forward pass of training
* 3 - The input gradient of training
* 4 - The filter gradient of training


EXECUTE
_______

This event contains the Poplar execution report in the ``execution_report``
field.

.. _using_the_ipu_model:

Using the IPU Model device for debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IPU Model is an emulator that mimics the IPU
computational framework on the host device. It is functionally equivalent to the
IPU, but obviously the compute performance will be completely different.

If you encounter an out of memory error, it may be useful to use the IPU Model
device to debug the problem.

Consider the situation in which the event trace is
being used to investigate a graph that creates a tile memory imbalance. In
this case, running on the IPU will lead to an out of memory exception
before the report is generated. Running on the IPU Model instead of actual
hardware will still run out of memory, but the code will run to completion
so the report can be generated.

There are a number of ways to target the IPU Model, but the simplest is to pass a flag to
TensorFlow using the ``TF_POPLAR_FLAGS`` environment variable. For example:

.. code-block:: console

    $ TF_POPLAR_FLAGS="--use_ipu_model" python basic_graph.py


See :ref:`env-var-section` for more information about this
environment variable.

.. code-block:: text

    ...] Device /device:IPU:0 attached to IPU: 0


where the "Device /device:IPU:0 attached to IPU: 0" indicates that the device
known to TensorFlow as "/device:IPU:0" is IPU 0.  The numbering of IPUs in your
machine can be found by using the ``gc-info -l`` command.

TensorFlow options for reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some tracing and reporting options are provided by TensorFlow as standard, and
can be useful when developing graphs for the IPU.

``TF_CPP_MIN_VLOG_LEVEL`` is an environment variable that enables the logging of
the main C++ backend.  Setting ``TF_CPP_MIN_VLOG_LEVEL=1`` will show a lot of
output.  Included in this is the compilation and execution of the IPU code.
The output of ``TF_CPP_MIN_VLOG_LEVEL`` can be overwhelming. ``TF_CPP_VMODULE``
provides a mechanism to reduce the logging to certain translation units (source
files).  This combination is quite useful:

.. code-block:: python

  TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1'

Finally, there is an environment variable called ``XLA_FLAGS`` which provides
options to the general XLA backend. For example, the follow will produce a
Graphviz DOT file of the optimised HLO
graph which is passed to the Poplar compiler.

.. code-block:: python

  XLA_FLAGS='--xla_dump_to=. --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re=forward-allocation --xla_hlo_graph_sharding_color'

The HLO pass ``forward-allocation`` is the final pass to run before the HLO
instructions are scheduled for passing to the Poplar graph compiler.
Running with these options will create a file
called something like
``module_0001.0001.IPU.after_forward-allocation.before_hlo-memory-scheduler.dot``.
(The way that the file names are generated is explained in :ref:`xla_file_naming`.)
The Graphviz ``dot`` command can be used to convert this data to an image.

More information on the XLA flags can be found in the definition of the XLA proto here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto


Reading the Poplar textual summary report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the example code is run, a new file is generated called
``Trace_Event_Report.rep``. This is the Poplar compilation report. The report is
broken into a number of sections, but here, we will focus on
the first three: **Target**, **Graph**, and **Memory Usage**.

Target
______

The "Target" section describes the target hardware which, in the absence of sharding, will be a
single IPU. For instance:

.. code-block:: text

    Target:
      Number of IPUs:         1
      Tiles per IPU:          1,216
      Total Tiles:            1,216
      Memory Per-Tile:        256.0 kB
      Total Memory:           304.0 MB
      Clock Speed (approx):   1,600.0 MHz


It is important to note that this section of the report does not distinguish
between hardware or the IPU Model, and in essence it is only dependent on the
number of IPUs selected for deployment via the sharding utility.

Graph
_____

The next section is "Graph", which describes the topology of the deployed
graph.

For instance:

.. code-block:: text


    Graph:
      Number of vertices:            1,219
      Number of edges:               1,223
      Number of variables:          30,562
      Number of compute sets:            4

You may see different numbers, depending on the version of the software.

This is from the report generated by the adder example. The graph map includes
control code, not just compute graph components. Note that the number of
vertices in the graph is very close to the 1,216 tiles on the IPU.

Memory usage
____________

The "Memory Usage" section gives the memory consumption profile of the graph
from a number of different perspectives:

.. code-block:: text


    Memory Usage:
      Total:
        Including Gaps:         23,878,396 B
        Excluding Gaps:
          By Memory Region:
            Non-interleaved:     5,355,604 B
            Interleaved:                 0 B
            Overflowed:                  0 B
          By Data Type:
              Variables:                            39,108 B
              Constants:                                 0 B
              Host Exchange Packet Headers:         10,512 B
              Global Exchange Packet Headers:            0 B
              Stack:                             3,852,288 B
              Vertex Instances:                     14,640 B
              Copy Descriptors:                          0 B
              VectorList Descriptors:                    0 B
              Vertex Field Data:                         0 B
              Control Table:                             0 B
              Control Code:                        851,272 B
              Vertex Code:                         170,788 B
              Internal Exchange Code:               60,792 B
              Host Exchange Code:                  351,328 B
              Global Exchange Code:                      0 B
              Instrumentation Results:               4,876 B
              Shared Code Storage:                       0 B
              Shared Data Storage:                       0 B
            Vertex Data (14,640B):
              By Category:
                Internal vertex state:          9,736 B
                Edge pointers:                  4,904 B
                Copy pointers:                      0 B
                Padding:                            0 B
                Descriptors:                        0 B
              By Type:
                poprand::SetSeedSupervisor                                                  34,048 B
                popops::ScaledAddSupervisor<float,float,true>                                   60 B
                popops::BinaryOp1DSupervisor<popops::expr::BinaryOpType::ADD,float>             16 B

      By Tile (Excluding Gaps):
        Range (KB) Histogram (Excluding Gaps)               Count (tiles)
             4 - 5 ****************************************  1,215
             5 - 6 *                                             1

        Maximum (Including Gaps): 49,184 (48.0 K) on tile 0
        Maximum (Excluding Gaps): 5,780 (5.6 K) on tile 0
        0 tile(s) out of memory


The information is presented in several sections. The first is the total
memory used, including gaps. This is followed by a breakdown of the
gap-excluding memory: first in terms of interleaved and non-interleaved usage,
then by data type, followed by vertex data.

A useful portion of the report is the tile histogram memory consumption
profile, which in this simple case is confined to two categories. When the
graph is more complex, the histogram will most likely have a more distributed
profile. In those instances, where there is in fact a tile imbalance, the
histogram produced may look more like this:

.. code-block:: text


    By Tile (Excluding Gaps):
        Range (KB) Histogram (Excluding Gaps)               Count (tiles)
           0 -   8 *                                            20
           8 -  16 ****************************************  1,192
          16 -  24 *                                             2
          24 -  32                                               0
          32 -  40                                               0
        .
        .
        .
         488 - 496                                               0
         496 - 504                                               0
         504 - 512 *                                             1
         512 - 520                                               0
         520 - 528                                               0
        .
        .
        .
         784 - 792                                               0
         792 - 800                                               0
         800 - 808                                               0
         808 - 816 *                                             1

        Maximum (Including Gaps): 834,416 (814.9 K) on tile 0
        Maximum (Excluding Gaps): 834,339 (814.8 K) on tile 0
        2 tile(s) out of memory


In this case, two tiles are out of physical memory, while most of the allocation
is well within the single tile budget.

In those instances where a memory imbalance
occurs, the report will produce a detailed description of the operations running
on five of the most memory-subscribed tiles (regardless of whether they are over their
physical limit or not) and list them in descending order of memory
consumption.

In the above case, tile 0 is the most over-subscribed tile, and
the report produces the following:

.. code-block:: text

    Tile # 0 memory usage:
    Memory Usage:
      Total:
        Including Gaps:            834,416 B
        Excluding Gaps:
          By Memory Region:
            Non-interleaved:       122,880 B
            Interleaved:           131,072 B
            Overflowed:            580,387 B
          By Data Type:
              Variables:                           807,658 B
              Constants:                                 0 B
              Host Exchange Packet Headers:          1,160 B
              Global Exchange Packet Headers:            0 B
              Stack:                                 3,168 B
              Vertex Instances:                     12,074 B
              Copy Descriptors:                      1,385 B
              VectorList Descriptors:                  960 B
              Vertex Field Data:                     7,934 B
              Control Table:                             0 B
              Control Code:                              0 B
                .
                .
                .

            Vertex Data (22,353B):
              By Category:
                Internal vertex state:          4,152 B
                Edge pointers:                 10,798 B
                .
                .
                .
              By Type:
                poplin::ConvPartial1x1Out<float,float,true,false>                                6,648 B
                poplar_rt::DstStridedCopy64BitMultiAccess                                        2,669 B
                popops::Reduce<popops::ReduceAdd,float,float,false,0>                            2,542 B
                popops::ScaledAddSupervisor<float,float,true>                                    1,440 B
                poplar_rt::StridedCopyDA32                                                       1,374 B
                poplar_rt::DstStridedCopyDA32                                                    1,101 B
                popops::BinaryOp1DSupervisor<popops::expr::BinaryOpType::MULTIPLY,float>           752 B
                .
                .
                .


This information can be very useful when tracking down the source of the
over-allocation.

Producing an ELF image of the compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is another method to produce much of the same detailed information
provided in the trace event report.
This generates code for IPU hardware (not an
emulator on the host) and then extracts the memory allocation information from the
generated ELF object file created at compile time. This technique will be described
briefly here, only showing how the object file is created and
memory-per-tile information extracted.

When compiling the graph, a Poplar engine option can be used to dump the ELF
file to a specified location.

.. code-block:: bash

    POPLAR_ENGINE_OPTIONS='{"target.saveArchive":"binaries.a", "debug.allowOutOfMemory": "true"}' python basic_graph.py


The file ``binaries.a`` is created, which is an archive file of the compiled graph.
To extract the memory size information from it, run the following command:

.. code-block:: console

    $ size -A binaries.a > tiles_elf.txt

This pipes a tile-by-tile rendition of the memory consumed in bytes to the file
``tiles_elf.txt``. All of the memory allocated is part of the text section.
This can be extracted from the tiles' ELF files to produce a single column where each entry
is the size of the text section corresponding to a tile:

.. code-block:: console

    $ size -A binaries.a | grep -e ".text" | awk '{print $2}' > memory_usage_per_tile.txt

The file ``memory_usage_per_tile.txt`` will contain this memory
allocation information. Further details of the deployed graph can be extracted with this
approach.

Dumping auxiliary Poplar information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two environment variable flags are available to get to extra Poplar
information: ``--save_vertex_graph`` and ``--save_interval_report``.

Poplar vertex graph
___________________

The Poplar vertex graph is a DOT file containing a complete description of the
lowered Poplar graph.  Each node in the graph represents one vertex in the
Poplar graph operating on one region of a tensor.

Poplar interval report
______________________

The interval report is a CSV file describing the number of tiles executing,
exchanging and syncing on each instruction cycle.

:ref:`env-var-section` describes how to set the environment flags.

.. xla_file_naming:

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


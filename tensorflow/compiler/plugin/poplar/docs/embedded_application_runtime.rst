IPU embedded application runtime
-------------------

The embedded application runtime allows you to run a compiled TensorFlow
executable as part of a TensorFlow graph. This enables embedding the executable
in a larger and more complex system, while also utilising IPUs. This runtime
appears in the TensorFlow graph as a custom CPU operation.

.. figure:: figures/embedded_app_graph.png
    :width: 100%
    :alt: Example of embedded application graph
    :align: center

    An arbitrary compute graph (left) with a possible IPU subgraph identified (middle), and a possible embedding of an executable (right)

The executable can be built with infeeds and outfeeds that maximise the
performance of the compiled application on the IPU. The feeds are presented to
the TensorFlow graph as standard inputs and outputs on the call operation.
These can be connected to other TensorFlow operations as part of a larger
graph.

Any mutable variables used inside the application are passed once in the call
to start. This minimises redundant communication with the IPUs.

Applications access this functionality through the
:py:func:`tensorflow.python.ipu.embedded_runtime.embedded_runtime_start` and
:py:func:`tensorflow.python.ipu.embedded_runtime.embedded_runtime_call` helper
functions.

Usage
~~~~~
The IPU embedded application runtime relies on instances of the
``RuntimeContext`` class to coordinate the startup and calls to the Poplar
engine. This object is created with a call to
:py:func:`tensorflow.python.ipu.embedded_runtime.embedded_runtime_start`.

.. code-block:: python

  from tensorflow.python.ipu import embedded_runtime

  ...

  context = embedded_runtime.embedded_runtime_start(
    poplar_exec_filepath, startup_inputs, engine_name)

The `startup_inputs` can be a list of tensors or a name-value dictionary of
tensors, where the names correspond to the name of the XLA inputs.

The created object is then passed to the call site where the
:py:meth:`tensorflow.python.ipu.embedded_runtime.embedded_runtime_call`
function can be called. The context object ensures all appropriate metadata is
passed, and control dependencies are created.

.. code-block:: python

  ...
  
  results = embedded_runtime.embedded_runtime_call(
    call_inputs, context)
  session.run(results)

Once the IPU embedded application runtime has been created and used within the
session, the Poplar engine will be running in a background thread. This thread
can outlive the TensorFlow session.

Example
~~~~~~~~
This example creates a very simple IPU program that doubles the input tensor.

.. literalinclude:: embedded_application_runtime_example.py
  :language: python
  :linenos:

Writing custom operations
-------------------------

If TensorFlow for the IPU does not implement an operation that you need then
there are two ways you can add a custom operation to the TensorFlow graph.

1.  You can implement the operation in C++ using the Poplar graph
    programming framework.
    See :numref:`custom_ipu_operation`.

    This provides the highest performance because the operation runs on the IPU.


2.  The second possibility is to execute the custom operation on the host CPU.
    See :numref:`custom_host_operations`.

    This may be easier to implement because you only need to write host code,
    without needing to get to grips with Poplar. However, the performance will
    be lower because it does not exploit the parallelism available on the IPU,
    and because the data has to be moved from the IPUs to the host and back.

.. note::

  In the rest of this chapter, "custom op" or "op" will be used to refer
  specifically to the new custom operation made available in the TensorFlow
  code. The word "operation" will be used more generally to talk about the
  implementation of this custom op.


.. _custom_ipu_operation:

Custom operation on the IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom op on the IPU, you need to write a Poplar program that
performs the required functions on the input tensors. After compiling this code,
you can load it into your TensorFlow program to create a custom op, which can then
be used in your TensorFlow model in the same way as any other op.

The following sections provide more detail on these steps.

Building the Poplar graph
_________________________

The custom op is defined in a C++ program that populates a graph with a
``poplar::Program`` object containing the operations to be performed on the
input tensors.
The Poplar and PopLibs libraries provide a rich set of functions optimised for
the IPU. You can also add your own functionality as "codelets", which contain
C++ code compiled for, and executed on, the IPU.

For more information about writing Poplar graph programs and codelets, refer to
the `Poplar and PopLibs User Guide
<https://docs.graphcore.ai/projects/poplar-user-guide/>`_ and the :tutorials-repo:`Poplar tutorials <tutorials/poplar>` on the Graphcore GitHub tutorials respository.

Your program must contain a function to build the graph, which will be called
from TensorFlow when you instantiate the custom op. This has the following
signature:

.. code-block:: cpp
  :linenos:

  extern "C"
  poplar::program::Program Build(
      poplar::Graph& graph,
      const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs,
      const std::string &attributes,
      const std::string &debug_prefix)

The default name for the function is ``Build()``. If you want to use a different
name (because you have multiple custom ops, for example), you can specify the name of
the function when importing the program into TensorFlow. See the definition of
the :py:func:`tensorflow.python.ipu.custom_ops.precompiled_user_op()` function
for details.

.. note::

  The ``extern "C"`` declaration is required to ensure that the compiler does
  not change the function name (C++ compilers will normally modify, or
  "decorate", function names to encode extra information about the function).

The parameters to ``Build()`` are:

* ``graph``: A Poplar graph to add the ``Program`` object and tensors to, in order to
  implement the operation.

* ``inputs``: A vector of tensors which are inputs to the operation.
  These are passed as the input arguments to the custom op when it is called in
  TensorFlow.

* ``outputs``: A vector of tensors that are the outputs of the operation. These
  will be returned as the result of the custom op in TensorFlow. This vector
  will initially be empty, so you will need to add result tensors to it.

* ``attributes``: A string which is passed as the ``attributes`` argument to
  the custom op in TensorFlow. See :ref:`operation_attributes` for more
  details.

* ``debug_prefix``: The debug name that is passed to the custom op in
  TensorFlow.

The ``Build()`` function returns the program object that it added to the graph.


Gradient builders
_________________

If the op is required for training, then you must also implement a function that
builds a Poplar graph for the gradient operation. This has the same name as the
forward-operation builder with ``_grad`` appended.

The signature of the gradient builder function is:

.. code-block:: cpp
  :linenos:

  extern "C"
  poplar::program::Program Build_grad(
      poplar::Graph& graph,
      int input_grad_index,
      const std::vector<poplar::Tensor>& gradients,
      const std::vector<poplar::Tensor>& fwd_inputs,
      const std::vector<poplar::Tensor>& fwd_outputs,
      std::vector<poplar::Tensor>& outputs,
      const std::string& attributes,
      const std::string& debug_prefix)


The parameters to ``Build_grad()`` are:

* ``graph``: A Poplar graph to add the ``Program`` object and tensors to, in order to
  implement the operation.

* ``input_grad_index``: The index of the input tensor to calculate the the partial derivative
  for.

  You can choose to implement a gradient operation that calculates the partial
  derivatives for all tensors or for one tensor at a time. In the latter case,
  you need to set ``separate_gradients`` to ``True`` when you call
  :py:func:`~tensorflow.python.ipu.custom_ops.precompiled_user_op()`.

  There may be advantages in calculating all the gradients at the same time; for
  example, if there are common sub-expressions. On the other hand, this removes
  the ability for TensorFlow to do some optimisations, such as dead-code
  elimination if all of the gradients are not required.

  If the ``separate_gradients`` parameter is set to ``False``, then your
  function for generating the gradient operation must populate one output tensor
  for each of the inputs of the forward pass function. Each output must be the
  partial derivative with respect to one of the inputs.

  If the ``separate_gradients`` parameter is ``True``, then the gradient
  operation building function must produce an operation with a single output,
  which is the partial differential with respect to only one of the forward pass
  inputs. The specific input will be given by the ``input_grad_index`` argument
  to the ``Build_grad()`` function.

  If your gradient operation calculates all of the partial derivatives, then you
  can ignore the ``input_grad_index`` parameter.

* ``gradients``: The inputs to the gradient operation, from the previous
  gradient operation or loss.

* ``fwd_inputs``: The input tensors to the forward-pass operation.

* ``fwd_outputs``: The output tensors from the forward-pass operation.

* ``outputs``: The outputs from this gradient operation. There must be one per
  input of the forward operation. Inputs which are not differentiable can be
  assigned a "null" Poplar tensor (that is, one created with the default
  ``Tensor`` constructor and containing no data).

* ``attributes``: A string which is passed as the ``gradient_attributes``
  argument to the custom op when called from TensorFlow. See
  :ref:`operation_attributes` for more details.

* ``debug_prefix``: The name of the operation.


The ``Build_grad()`` function returns the program object that it added to the graph.


.. _custom_op_metadata:

Metadata
________

You can also specify extra information about the custom op by including a
*metadata* function in the object file. This has the same name as the builder
function with ``_metadata`` appended.

This function has the following signature:

.. code-block:: cpp
  :linenos:

  extern "C"
  void Build_metadata(
      std::vector<std::int64_t>& allocating_indices,
      std::vector<std::int64_t>& replica_identical_output_indices,
      std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
      bool& is_elementwise,
      bool& is_stateless,
      bool& is_hashable,
      std::uint32_t num_inputs)

The parameters are used to return the following information about the operation:

* ``allocating_indices``: Use this to specify which input tensors will
  be allocated using the tensor-allocation function described in
  :numref:`tensor_allocation`.

* ``replica_identical_output_indices``: Experimental. Use this to specify which
  output tensors are identical across replicas. The compiler uses this to help
  provide deterministic behaviour when running with replication and performing
  stochastic rounding.

  An empty vector means that no tensors are identical across replicas.

.. _input_to_output_tensor_aliasing:

* ``input_to_output_tensor_aliasing``: Use this map to indicate if any of the
  input and output tensors alias. The values in the map are the vector indexes
  of the the tensors. For example, a mapping from 1 to 0 indicates that
  input tensor 1 is aliased with output tensor 0. This means that
  ``poplar::Tensor::intersectsWith()`` would return true when called for these
  tensors.

  Providing information about whether an input tensor aliases an output tensor
  allows the TensorFlow graph compiler to perform more optimisation. It also ensures
  that if an input tensor is updated in-place and used as an output, then any
  other uses of that tensor will be completed before this operation is run, to
  ensure correct behaviour. See :ref:`inplace_operations` for an example of
  using this for an in-place operation.

  If an input tensor is *not* mapped to an output tensor, then the operation
  must not modify that input tensor. If it is modified, then other operations
  which use it as an input may be passed incorrect values.

* ``is_elementwise``: Set this to true if the output of an operation is the same
  shape and layout as its first input. (This parameter was originally used to
  tell the compiler that an operation was elementwise. However, its meaning has
  changed to indicate any operation where the compiler can perform optimisations
  based on matching the input and output tensors.)

  In this case, your graph-building code for the operation will typically clone the input in
  order to generate the output tensor.

* ``is_stateless``: Set this to true if this operation is "stateless".

  If an operation's outputs depend only on the value of their inputs, and not
  any internally stored state, then the operation is said to be stateless.
  Marking an operation as stateless will allow the TensorFlow backend to perform
  optimisations which would otherwise not be possible, such as common code
  removal. It also allows the custom op to be used with recomputation, see
  :numref:`recomputation`.

  Custom ops are stateful by default.

* ``is_hashable``: Set this to true if this operation can be uniquely hashed.

  In order to detect when code changes and needs to be recompiled, the
  TensorFlow compiler will generate a hash value for the TensorFlow graph. If
  all ops in the graph are hashable then the executable will be saved in the
  cache (if enabled). This allows the graph to be run multiple times without
  needing to recompile it. See :numref:`caching_executables` for more information.

  However, because the TensorFlow compiler does not
  have any information about the implementation of the custom operation or its
  dependencies, the compiler will treat it as non-hashable, therefore the
  TensorFlow program will be recompiled every time it is run.

  If you can guarantee that custom operation and its dependencies will not
  change then you can set this parameter to true.

  This attribute must be set to true if you intend to pre-compile your TensorFlow program
  (see :numref:`precompiling_executables`).

* ``num_inputs``: This is the number of input tensors that the operation is called with.

If you use the metadata function to specify some information about the custom
operation, then you must set the values of *all* the parameters even if you are
using the default values.

Gradient builders have their own metadata functions. These are named after the
gradient builder function with ``_metadata`` appended. For example:
``Build_grad_metadata()``.


Compiling the IPU code
______________________

API level
.........

You need to specify the API level that your operation code is compatible
with. The custom op loader checks the API level and will not load it if it does
not match the current API level. A change in API level normally means that the
file is not compatible with previous versions. See
:numref:`api_level_changes` for information about the changes in the API.

You must include the following code in your builder program to specify the API
level.

.. literalinclude:: custom_rotate_op.cc
  :linenos:
  :language: C++
  :start-at: // Export the API level symbol
  :end-at: }

.. list-table:: API level changes
  :align: center
  :width: 100%
  :widths: 20, 80
  :header-rows: 1
  :name: api_level_changes

  * - API level
    - Changes to the API

  * - 1
    - ``is_stateless`` was added to the metadata function.

  * - 2
    - The ``attributes`` parameter was added to the allocation and the
      build functions to allow user-defined attributes to be passed to the
      operation (and its gradient operation, if present).

  * - 3
    - ``input_to_output_tensor_aliasing`` replaced ``num_inplace`` to allow
      finer-grain description of the operation performed in order to allow more
      optimisations.

  * - 4
    - ``is_hashable`` was added to the metadata builder function.

  * - 5
    - ``replica_identical_output_indices`` was added to the metadata builder function.

PopLibs library code
....................

You need to explicitly add the the IPU code for any PopLibs libraries that you use.
For example, if your code uses the ``popops`` and ``poprand``
libraries, then you need to include the following in your builder code:

.. code-block:: cpp
  :linenos:

  #include <popops/codelets.hpp>
  #include <poprand/codelets.hpp>

  extern "C"
  poplar::program::Program Build_grad(poplar::Graph& graph,
                                      int input_grad_index,
                                      const std::vector<poplar::Tensor>& gradients,
                                      const std::vector<poplar::Tensor>& fwd_inputs,
                                      const std::vector<poplar::Tensor>& fwd_outputs,
                                      std::vector<poplar::Tensor>& outputs,
                                      const std::string& attributes,
                                      const std::string& debug_prefix) {

      ... // create the program object in the graph

      popops::addCodelets(graph);
      poprand::addCodelets(graph);
  }


Compiling the library file
..........................

The code has to be compiled to create a shared-library object file.
For example, if you have a source file called ``poplar_code.cpp`` that contains
the ``Build()`` function, you can use the following command line to generate
a library file called ``libcustom_op.so``:

.. code-block:: console

  $ g++ poplar_code.cpp -shared -fpic -o libcustom_op.so -lpoplar -lpoputil -lpoprand

Note that you need to link the Poplar and PopLibs libraries that you use
(in this example ``poplar``, ``poputil`` and ``poprand``). See the `Poplar and
PopLibs API Reference
<https://docs.graphcore.ai/projects/poplar-api/page/using_libs.html>`_ for more
information.

It is not necessary to include or link against any TensorFlow header or library
files. Only the Poplar and PopLibs headers, and the corresponding libraries are
required.

You can add ``-g`` to the above command to compile the custom operation with
debugging symbols. This allows you to debug the C++ code with ``gdb``. 


Using the custom op in TensorFlow
__________________________________


You can call the custom operation from TensorFlow with
:py:func:`~tensorflow.python.ipu.custom_ops.precompiled_user_op()`. This
specifies the library file containing the custom operation code, the input and
output tensors, and other information needed to use the op in TensorFlow. See
:py:func:`~tensorflow.python.ipu.custom_ops.precompiled_user_op()` in the API documentation for more
information.


.. _tensor_allocation:

Tensor allocation
_________________

If the input tensors to the operation have not already been allocated to tiles
because of their use by other operations, then the TensorFlow compiler will, by default,
allocate the tensors with linear mapping.

You can override this behaviour by defining a function that allocates
tensors in a way that is most efficient for your operation.
See the section on variable mapping in the `Poplar and PopLibs API Reference
<https://docs.graphcore.ai/projects/poplar-api/page/poplar_api/poplar/graph/VariableMappingMethod.html>`_
for moe information.

To do this, define a function with the suffix ``_allocator`` with the following
signature:

.. code-block:: cpp
  :linenos:

  extern "C" poplar::Tensor Build_allocator(
      poplar::Graph& graph,
      std::uint32_t operand,
      const std::vector<size_t>& shape,
      poplar::Type type,
      const std::string& attributes,
      const std::string& debug_prefix)

The parameters to the function are:

* ``graph``: The graph to add the tensor to.
* ``operand``: The index of the input tensor to allocate.
* ``shape``: The shape of the tensor.
* ``type``: The Poplar data type for the tensor.
* ``attributes``: A string which is passed as the ``attributes`` or
  ``gradient_attributes`` argument to the custom op in TensorFlow (depending on whether
  this function corresponds to the forward or gradient operation). See
  :ref:`operation_attributes` for more details.
* ``debug_prefix``: the name of the operation.

The allocator function returns the tensor that it has allocated.

If the input tensor has already been allocated, then this function will
not be called.

.. _ipu_examples:

Examples
________

Some examples of using a custom op in TensorFlow are shown in the following
sections. There are further examples in the Graphcore GitHub tutorials
repository:

* :tutorials-repo:`Custom op with gradient <feature_examples/tensorflow/custom_gradient>`

* :tutorials-repo:`Custom op with codelet for custom vertex code <feature_examples/tensorflow/custom_op>`


.. _inplace_operations:

In-place operations
...................

An operation can use the same tensor as an input and output, modifying the
tensor in-place as opposed to creating a new output tensor.

You can use the :ref:`input_to_output_tensor_aliasing
<input_to_output_tensor_aliasing>` map in the metadata to indicate this to the
TensorFlow compiler by specifying that the input tensor is aliased with an
output tensor.

When you update tensors in-place, the TensorFlow compiler must see an assignment
to the tensor, otherwise the changes to the input tensor will be optimised away.
This means that the in-place inputs always need to be returned as outputs of the
custom operation. If a ``tf.Variable`` object is modified in-place then it has to be
assigned back to itself with ``tf.assign``.

:numref:`custom_add_inplace_cc` shows an example of adding an in-place custom op to
a TensorFlow model.
The implementation of the operation is shown in :numref:`custom_add_inplace_cc`.

.. literalinclude:: custom_add_inplace.py
  :language: python
  :linenos:
  :caption:
  :name: custom_add_inplace_py
  :start-at: import

.. only:: html

  Download :download:`custom_add_inplace.py`


.. literalinclude:: custom_add_inplace.cc
  :language: cpp
  :linenos:
  :caption:
  :name: custom_add_inplace_cc
  :start-at: #include

.. only:: html

  Download :download:`custom_add_inplace.cc`


.. _operation_attributes:

Operation attributes
....................

If an operation requires some data which is not available when compiling the C++
builder function, then the string ``attributes`` argument can be used to pass
such information from the TensorFlow op to the C++ function.
Since the ``attributes`` argument is a string object, any data format which can
be serialized/deserialized as a string, such as JSON, can be used.

In :numref:`tutorial_attributes_example_cc`, we implement a custom operation which
performs a serialized matrix-matrix multiplication where the ``attributes``
argument passes information about serialization, encoded in JSON data format, to
the C++ function. :numref:`tutorial_attributes_example_py` shows how this custom
op is called from TensorFlow.

.. literalinclude:: tutorial_attributes_example.cc
  :language: C
  :linenos:
  :caption:
  :name: tutorial_attributes_example_cc
  :start-at: #include

.. only:: html

  Download :download:`tutorial_attributes_example.cc`


.. literalinclude:: tutorial_attributes_example.py
  :language: python
  :linenos:
  :caption:
  :name: tutorial_attributes_example_py
  :start-at: import

.. only:: html

  Download :download:`tutorial_attributes_example.py`


Custom codelet
..............

:numref:`custom_rotate_op_cc` shows the source file for a custom rotate
operation, which takes three vectors and rotates ``x`` and ``y`` by the values
in ``angle``. The vertex code for the custom codelet is shown in
:numref:`custom_codelet_cpp`. The TensorFlow program that calls the custom op is
shown in :numref:`tutorial_custom_codelet_py`.


.. literalinclude:: custom_rotate_op.cc
  :language: C
  :linenos:
  :caption:
  :name: custom_rotate_op_cc
  :start-at: #include

.. only:: html

  Download :download:`custom_rotate_op.cc`


.. literalinclude:: custom_codelet.cpp
  :language: cpp
  :linenos:
  :caption:
  :name: custom_codelet_cpp
  :start-at: #include

.. only:: html

  Download :download:`custom_codelet.cpp`


.. literalinclude:: tutorial_custom_codelet.py
  :language: python
  :linenos:
  :caption:
  :name: tutorial_custom_codelet_py
  :start-at: import

.. only:: html

  Download :download:`tutorial_custom_codelet.py`


.. _custom_host_operations:

Custom host CPU operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can write a custom operation as a function that executes code on the host
CPU instead of on the IPU. The default name for this function is ``Callback()``.
As with the builder functions described previously, this must be compiled into a
shared library file.

The signature of the callback function is:

.. code-block:: C++
  :linenos:

  extern "C"
  void Callback(
      const std::vector<const void*>& data,
      const std::vector<std::uint32_t>& number_of_elements,
      const std::vector<void*>& outputs,
      const std::string& attributes,
      const std::string& name);

The parameters are:

* ``data``: The input data passed to the custom op in TensorFlow. The function
  must be written to expect a specific data type and the void pointer cast
  into the expected type.

* ``number_of_elements``: This indicates the number of elements in the input
  data.

* ``outputs``: The results returned by the operation.

* ``attributes``: A string which is passed as the ``attributes`` argument to
  the custom op in TensorFlow. See :ref:`operation_attributes` for more
  details.

* ``name``: This is the name of the operation within the XLA graph.

You can call the host code from your TensorFlow program using
:py:func:`tensorflow.python.ipu.custom_ops.cpu_user_operation()`.
This specifies the input object file to load, the input and output tensors, and other parameters to the
operation.

Gradient callback
_________________

If the op is required for training, then you must also implement a function for
the gradient operation. This has the same name as the callback with ``_grad``
appended.

The signature of the gradient callback function is:

.. code-block:: C++
  :linenos:

  extern "C" void Callback_grad(
      const std::vector<void*>& data,
      const std::vector<uint32_t>& number_of_elements,
      std::vector<void*>& outputs,
      const std::string& attributes,
      const std::string& name);

The parameters are:

* ``data``: The input data passed to the custom op in TensorFlow. The function
  must be written to expect a specific data type so the void pointer can be cast
  into the expected type.

* ``number_of_elements``: This indicates the number of elements in the input data.

* ``outputs``: The results returned by the operation.

* ``attributes``: A string which is passed as the ``gradient_attributes``
  argument to the Python op in TensorFlow. See :ref:`operation_attributes` for
  more details.

* ``name``: This is the name of the operation within the XLA graph.

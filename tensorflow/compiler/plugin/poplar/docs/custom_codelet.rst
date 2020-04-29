Custom IPU operations
---------------------

There are three mechanisms for providing custom operations to the IPU through
the TensorFlow interface.  The first uses a fully custom codelet and host
build file.

The second case is a custom operation which is executed on the CPU.

The third possibility is a custom, fused elementwise arithmetic operation. In this last
case, the gradient creation in the optimisers will not produce a gradient
operation for the custom operation.

Fully customised IPU operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can provide a custom operation to be compiled into the Poplar
executable and run on the IPU hardware. You must provide a host-side
shared object library that implements the action of adding vertices to a
Poplar graph, given some Poplar tensor inputs.  They can optionally provide
a Poplar source code or binary file containing one or more "codelets"
(code that runs on the IPU).

For more information about writing codelets, please refer to the
`Poplar and Poplibs User Guide
<https://documents.graphcore.ai/documents/UG1/latest>`_.

These operations are added with ``ipu.user_ops.precompiled_user_op``. More
information about this can be found in :ref:`api-section`.  An example of
this is shown below.

The shared object file must contain an undecorated symbol, that should be
declared as below.  It should add vertices to the graph that perform the
custom operation.  The name of the symbol should match the name of the
operation in the graph.  By default these types of operations are called
``Build``.

.. code-block:: cpp

  extern "C"
  poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string &debug_prefix)

The arguments are:

* ``graph``: the Poplar graph into which to add tensors and vertices.

* ``inputs``: a vector of Poplar tensors which are inputs to the operation.

* ``outputs``: a vector into which to store the outputs of the operation. The
  vector will contain zero entries when the ``Build`` function is called.

* ``debug_prefix: the debug name that has been given to the operation in
  the TensorFlow graph.

If the operation can have its gradient taken, then the shared object can
contain a separate function with the same name as the forward pass builder.
The function must be given the same name as the forward operation with ``_grad``
appended.  The signature of the builder function is slightly different, as it
takes the forward pass inputs and outputs as arguments, as well as the
gradient outputs.

.. code-block:: cpp

  extern "C"
  poplar::program::Program Build_grad(
      poplar::Graph& graph, int input_grad_index,
      const std::vector<poplar::Tensor>& gradients,
      const std::vector<poplar::Tensor>& fwd_inputs,
      const std::vector<poplar::Tensor>& fwd_outputs,
      std::vector<poplar::Tensor>& outputs,
      const std::string& debug_prefix)

The arguments are:

* ``graph``: the Poplar graph into which to add tensors and vertices.

* ``input_grad_index``: The index of the input for which this operation is producing
  the partial derivative.  If the gradient operation
  calculates all of the partial derivatives, then this input
  should be ignored.

* ``gradients``: the inputs to the gradient operation, from the previous gradient operation
  or loss.

* ``fwd_inputs``: the tensors which are the inputs to the forward operation.

* ``fwd_outputs``: the tensors which are the outputs of the forward operation.

* ``outputs``: the outputs of this gradient operation. There must be one per
  input of the original forward operation.  Inputs which are not
  differentiable can have an null Poplar tensor.

* ``debug_prefix``: the name of the operation.

Metadata
________

The shared object file can optionally contain an undecorated symbol that is
the same as the builder function with ``_metadata`` appended.  This function
must have the following signature:

.. code-block:: cpp

  extern "C"
  void Build_metadata(std::vector<std::int64_t>& allocating_indices,
    std::uint32_t& num_inplace, bool& is_elementwise,
    std::uint32_t num_inputs)

The arguments are:

* ``allocating_indices``: indicates which of the inputs should be allocated
  using the tensor allocation function.  See the
  description in :ref:`tensor_allocation`.

* ``num_inplace``: indicates the number of inputs which are 'in place'.  The first
  ``num_inplace`` of the inputs will be considered to be in-place.

* ``is_elementwise``: indicates that this operation is element-wise.

* ``num_inputs``: indicates how many inputs are on the operation.

The function should fill in the values of the first three arguments, which
are all reference types.


In-place operations
___________________

If an operation does an in-place modification of an input tensor, as
opposed to creating a new output tensor, then the ``num_inplace`` can be
used to indicate that this is the case.  The system will ensure that when
a tensor is updated in place, that any other uses of that tensor will be
complete before the operation is run.

If a tensor is not marked as `in place` then the operation must not modify
it.  If it is modified then other operations which consume it may see an
incorrect value on their input.

Elementwise operations
______________________

The IPU driver can do a better job of allocating the layout of Poplar tensors
if it can associate them with specific operations.  If the output of an
operation is the same shape and layout as its first input, then it should be
marked as elementwise.

Typically, the graph building code for the operation will clone the input in
order to generate the output Poplar tensor.

.. _tensor_allocation:

Tensor allocation
_________________

When generating the Poplar graph, sometimes the backend has the freedom to
allocate an input to an operation.  This happens when an input to an operation is
also the input to the graph, or when previous operations do not put
constraints on the input tensor.

If this condition occurs, then by default the backend will create the Poplar
tensor with linear mapping.  See the section on tile mapping in the
`Poplar and Poplibs API Reference
<https://documents.graphcore.ai/documents/UG2/latest>`_.

To override this behaviour and allocate a tensor using a specific layout
mapping, the custom operation can provide a function with the following
signature:

.. code-block:: cpp

  extern "C" poplar::Tensor Build_allocator(
    poplar::Graph& graph, std::uint32_t operand,
    const std::vector<size_t>& shape, poplar::Type type,
    const std::string& debug_prefix)

The arguments are:

* ``graph``: the Poplar graph where the tensor should be created.
* ``operand``: the operand number of the input to allocate.
* ``shape``: the shape of the tensor.
* ``type``: the Poplar data type for the tensor.


Gradient operations
___________________

As described above, when the gradient of the forward operation is generated,
either a single operation, or multiple operations can be inserted into the graph.

You can use the parameter ``separate_gradients`` on the ``precompiled_user_op`` function
to select which of the two options are required.  The
compiled code must match this setting.

If the ``separate_gradients`` parameter is set to ``False``, then the compiled
function for generating the gradient operation should fill in one output
for each of the inputs of the forward pass function.  Each output should be
the partial derivative with respect to one of the inputs.

If the ``separate_gradients`` parameter is ``True``, then the gradient operation
building function should produce an operation with a single output, which is
the partial differential with respect to only one of the forward pass inputs.

The specific input will be given by the ``input_grad_index`` input of the call
to the sharded object ``Build_grad`` function.

Example
_______

This example shows the source file for a rotate operation, which takes three vectors
and rotates the ``x`` and ``y`` ones by the ``angle`` one:

.. literalinclude:: custom_rotate_op.cc

This is the associated codelet file:

.. literalinclude:: custom_codelet.cpp

This is an example of it in use:

.. literalinclude:: tutorial_custom_codelet.py

When compiling the host-size shared object file, it is not necessary to
include or link against any TensorFlow header or library files.  Only the
Poplar headers and link libraries should be necessary.

Fully customised CPU operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework also allows a custom operation that executes code on the CPU
instead of on the IPU.  A shared object, much like the builder function of
the device-side custom operation, must be written.  The signature of this
function should be:

.. code-block:: C++

  extern "C" void Callback(const std::vector<void*>& data,
                           const std::vector<std::uint32_t>& number_of_elements,
                           std::vector<void*>& outputs,
                           const std::string& name);

The arguments are:

* ``data``: the input data. the function should be written to expect a certain data
  type so the void pointer can be cast into the expected type.
* ``number_of_elements``: indicates the number of elements in the input data.
* ``outputs``: should be filled in by the operation.
* ``name``: is the name of the operation within the XLA/HLO graph.

Custom elementwise expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python class ``ipu.custom_ops.codelet_expression_op`` provides an interface
for giving a custom fused expression to the compiler.  This will be encoded
into a single compute set.

The arguments to the Python function are a callable Python function which
encodes the arithmetic expression, and the tensor arguments to the operation.

For instance:

.. code-block:: python

  def my_custom_op(x, y, z):
      return x * x + y * z

  ipu.custom_ops.codelet_expression_op(my_custom_op, a, b, c)

In this example, the Python function ``my_custom_op`` provides the expression,
and the arguments ``a``, ``b`` and ``c`` are the three inputs from other parts of
the TensorFlow graph.

Python operators which are supported in the function are ``+``, ``-``, ``*``, and
``abs``.

API Level Versioning
~~~~~~~~~~~~~~~~~~~~

.. code-block:: C++

  extern "C" int32_t Build_api_level;

This is reserved for the future changes in API which may render current binary
modules incompatible with current code. Loader checks its value and refuses to
load modules with different API level.

Current default value is 0, so it's not necessary to specify api_level of 0,
but this might change in the future.

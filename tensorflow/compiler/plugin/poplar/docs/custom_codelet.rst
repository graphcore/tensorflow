Custom IPU operations
---------------------

There are two mechanisms for providing custom operations to the IPU through the
TensorFlow interface.  The first allows a fully custom codelet and host build
file.  The second allows a custom fused elementwise arithmetic operation.

In both of these cases, the gradient creation in the Optimizers will not
produce a gradient operation for the custom operation.

Fully customized operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A user can provide a custom operation that can be compiled into the Poplar
executable and run on the IPU hardware.  The user must provide a host side
shard object library which implements the action of adding vertices to a
Poplar Graph, given some Poplar Tensor inputs.  They can optionally provide
a Poplar `GP` file containing one or more codelets.

The shared object file must contain an undecorated symbol `Build`, which should
be declared as below.  It should add vertices to the graph that perform the
custom operation.

::

  extern "C"
  poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string &debugPrefix)

The arguments are:

* graph - the poplar graph into which to add tensors and vertices.
* inputs - a vector of poplar tensors which are inputs to the operation.
* outputs - a vector into which to store the outputs of the operation. The
            vector will contain zero entries when the `Build` function is
            called.
* debugPrefix - the debug name that has been given to the operation in the
                TensorFlow graph.

The shared object file can optionally contain an undecorated symbol
`IsElementWise`, which indicates whether the custom operation is element-wise.
If an operation takes one or more tensors of the same shape, and performs an
expression on only corresponding elements in the input tensors,  and produces
a tensor of the same shape, then it is elementwise.

::

  extern "C" bool IsElementWise()


This example shows the source file for a rotate op, which takes three vectors
and rotates the `x` and `y` ones by the `angle` one.

.. literalinclude:: custom_rotate_op.cc

This is the associated codelet file.

.. literalinclude:: custom_codelet.cpp

This is an example of it in use:

.. literalinclude:: tutorial_custom_codelet.py

When compiling the host-size shared object file, it is not necessary to include
or link against any TensorFlow header or library files.  Only the Poplar
headers and link libraries should be necessary.

Custom elementwise expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The python class `ipu.custom_ops.codelet_expression_op` provides an interface
for giving a custom fused expression to the compiler.  This will be encoded into
a single compute set.

The arguments to the python function are a callable python function which
encodes the arithmetic expression, and the tensor arguemnts to the operation.

For instance:

::

  def my_custom_op(x, y, z):
      return x * x + y * z

  ipu.custom_ops.codelet_expression_op(my_custom_op, a, b, c)

In this example, the python function `my_custom_op` provides the expression,
and the arguments `a`, `b` and `c` are the three inputs from other parts of the
TensorFlow graph.

Python operators which are supported in the function are `+`, `-`, `*`, and
`abs`.


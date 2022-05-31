IPU Outlined Functions
----------------------
An outlined function is a block of organized, reusable code which is used to
perform a single action. Functions provide better modularity for your
application and a high degree of code reusing which can decrease the memory
usage as only one copy of the code needs to be compiled. Using functions however
can increase the amount of computations as the function inputs need to be copied
to the correct function argument locations and the function outputs need to be
returned as well.

If the provided function contains any stateful operations, such as stateful
random number generation, then the function cannot be reused and it will be
inlined automatically.

Note that the function code is only reusable for calls on the same IPUs. This
means that benefits of function calls will only be seen if the function calls
are made from the same shard, or a pipeline stage mapped to the same IPU.

IPU outlined functions should not be confused with `tf.function` which creates a
TensorFlow graph, whereas the IPU function creates a Poplar function which can
be used inside of `tf.function`.

Usage
~~~~~
The Python function provided can only take a list of positional arguments. All
of the arguments must be `tf.Tensor`-like objects, or be convertible to them
(for example constants).
Other non `tf.Tensor`-like objects can still be accessed by the function using
Python closure capturing.

IPU functions can access TensorFlow variables, however unless each function
invocations is meant to use the same variable, a `variable_scope` should be
used.

A `variable_scope` is not a `tf.Tensor`-like object and therefore it cannot be
passed as an argument, so if we used the following function:

.. literalinclude:: function_block1.py
  :language: python
  :linenos:

Each invocation of the function of the function will use the same variable.

To circumvent this, we can use Python closures to create unique scopes for each
invocation of the function:

.. literalinclude:: function_block2.py
  :language: python
  :linenos:

Here we wrap the IPU function (`f`) in a Python function(`func`), which has
extra arguments (the variable scope name). These extra arguments can then be
captured by the IPU function `f` resulting, meaning that each invocation of the
function will result in different variables being captured.

Alternatively we can explicitly pass the `tf.Variables` as inputs to the
function:

.. literalinclude:: function_block3.py
  :language: python
  :linenos:

Examples
~~~~~~~~

Functions can be beneficial in many scenarios, especially where we want to
reduce the amount of code generated.

Models with common structures
_____________________________

Some models often have common structures/layers residing on the same IPU,
where the inputs and outputs have the same shapes and data types. We can create
a single function for these common building blocks to reduce the code size.

.. literalinclude:: function_example1.py
  :language: python
  :linenos:

Download :download:`function_example1.py`

Serializing large operations
____________________________
Some operations in the model might generate large intermediate values which can
cause large spikes in memory usage.
Such spikes can be reduced by serializing the operation, however it can result
in extra code. To try and avoid the extra code, IPU functions can be used.

.. literalinclude:: function_example2.py
  :language: python
  :linenos:

Download :download:`function_example2.py`


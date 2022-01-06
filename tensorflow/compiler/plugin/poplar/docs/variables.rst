Adding variables
----------------

Do not add variables using ``tf.Variable([shape], initializer)``, because they will fail
to obey certain operations, such as ``assign_add``.

Make sure that all variables are added using a variable scope that is marked as
a resource. This can be done globally, as shown below:

.. code-block:: python

  vscope = tf.get_variable_scope()
  vscope.set_use_resource(True)
  ...
  var = tf.get_variable(name, shape=[...], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
  ...

Or it can be done locally, in a specific scope:

.. code-block:: python

  with tf.variable_scope("vs", use_resource=True):
    var = tf.get_variable(name, shape=[...], dtype=tf.float32, initializer=tf.constant_initializer(0.5))

Troubleshooting
~~~~~~~~~~~~~~~

If you get an error similar to the following (especially the lines containing
``VariableV2``) it indicates that a variable has been created which is not a
resource variable.

.. code-block:: none

    InvalidArgumentError (see above for traceback): Cannot assign a device for operation
      'InceptionV1/Logits/Conv2d_0c_1x1/biases': Could not satisfy explicit device specification
      '/device:IPU:0' because no supported kernel for IPU devices is available.
    Colocation Debug Info:
    Colocation group had the following types and devices:
    Const: CPU IPU XLA_CPU
    Identity: CPU IPU XLA_CPU
    Fill: CPU IPU XLA_CPU
    Assign: CPU
    VariableV2: CPU

Note on the global_step counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More advanced execution control frameworks in TensorFlow use a scalar counter
called ``global_step`` to count the number of iterations of training which have
occurred. This counter is serialised along with the model. It allows the model
to base parameters on the step count, even if the model is run multiple times.

There is an ``add`` operation which adds to the ``global_step`` scalar on each
training pass.  If the ``global_step`` variable is placed on the IPU device,
then this increment operation will occur on the IPU too.  This will cause the
Poplar training engine to be swapped out for the increment engine on each
training step, causing very poor performance.

To avoid this, in the CPU context, use the expression
``tf.train.get_or_create_global_step()`` before you create any special training
sessions.  This will ensure that the global_step variable is on the CPU.

.. code-block:: python

  with tf.device("cpu"):
    tf.train.get_or_create_global_step()

  with ipu.scopes.ipu_scope("/device:IPU:0"):
    out = ipu.ipu_compiler.compile(model_fn, [...])

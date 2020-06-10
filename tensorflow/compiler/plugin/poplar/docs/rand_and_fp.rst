Half-precision floating point and stochastic rounding
-----------------------------------------------------

The IPU supports IEEE half-precision floating-point numbers, and supports
stochastic rounding in hardware.  The IPU extensions to TensorFlow expose this
floating point functionality through the functions described below.
See the :ref:`api-section` for more detail.

Controlling the half-precision floating-point unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure the behaviour of the floating-point hardware using the function
:py:func:`tensorflow.python.ipu.utils.set_floating_point_behaviour_options`.

The ``esr`` bit enables the stochastic rounding unit. Three of the remaining
options control the generation of hardware exceptions on various conditions.
The ``nanoo`` bit selects between clipping or generating a NaN
when a half-precision number overflows.

Resetting the global random number seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stochastic rounding unit and the TensorFlow stateful random number
generators both use a common global random number seed to initialise the
random number generator hardware. Each IPU device has its own seed.

By default this seed is set randomly, but it can be reset by using the function
:py:func:`tensorflow.python.ipu.utils.reset_ipu_seed`.

Due to the hardware threading in the device, if the seed reset function is used
then the ``target.deterministicWorkers`` Poplar Engine option will need to be set
to ``true``.

This can be done using
:py:func:`tensorflow.python.ipu.utils.set_compilation_options`.

Debugging numerical issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

The values held in a tensor can be printed by calling ``ipu.internal_ops.print_tensor``.
This function takes a tensor and will print it to standard error as a side
effect.

See :py:func:`tensorflow.python.ipu.internal_ops.print_tensor`.

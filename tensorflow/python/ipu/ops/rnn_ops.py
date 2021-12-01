# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Popnn recurrent neural network operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation

POPNN_LSTM = "lstm"
POPNN_GRU = "gru"
POPNN_DYNAMIC_GRU = "dynamic_gru"
POPNN_AUGRU = "augru"

POPNN_LSTM_NUM_GATES = 4
POPNN_GRU_NUM_GATES = 3
POPNN_DYNAMIC_GRU_NUM_GATES = 3
POPNN_AUGRU_NUM_GATES = 3

__all__ = ["PopnnLSTM", "PopnnGRU", "PopnnDynamicGRU", "PopnnAUGRU"]


class _PopnnRNN(base_layer.Layer):  #pylint: disable=W0223
  """Base class for implementing XLA and Popnn compatible RNN layers.
  """
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a _PopnnRNN model from model spec.

    Args:
      num_units: the number of units within the RNN model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weight
        (default is all zeros).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      available_memory_proportion_fwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the forward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used. If available_memory_proportion_bwd is set to None,
        then this value applies to both phases.
      available_memory_proportion_bwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the backward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used.
    """
    super(_PopnnRNN, self).__init__(dtype=dtype, name=name)

    if available_memory_proportion_bwd is None:
      available_memory_proportion_bwd = available_memory_proportion_fwd

    if dtype not in [dtypes.float16, dtypes.float32]:
      raise ValueError("Only support float16, float32, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._plain_dtype = dtype
    self._partials_dtype = partials_dtype
    self._num_layers = 1
    self._num_units = num_units
    self._weights_initializer = weights_initializer
    self._bias_initializer = bias_initializer
    self._seed = seed
    # Init input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

    activation = op_util.get_activation_name(activation)
    recurrent_activation = op_util.get_activation_name(recurrent_activation)

    self._activation = activation
    self._recurrent_activation = recurrent_activation
    self._available_memory_proportion_fwd = available_memory_proportion_fwd
    self._available_memory_proportion_bwd = available_memory_proportion_bwd

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def num_units(self):
    return self._num_units

  @property
  def input_size(self):
    if not self._input_size:
      raise ValueError(
          "\'input_size\' is unknown since layer has not been built.")
    return self._input_size

  @property
  def saveable(self):
    raise NotImplementedError(
        "This cell does not yet support object-based saving. File a feature "
        "request if this limitation bothers you.")

  @property
  def canonical_weight_shape(self):
    """Shapes of Popnn canonical weight tensors."""
    if not self._input_size:
      raise RuntimeError(
          "%s.canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    return self._canonical_weight_shape(0)

  @property
  def canonical_bias_shapes(self):
    """Shapes of Popnn canonical bias tensors."""
    return self._canonical_bias_shape(0)

  def build(self, input_shape):
    raise ValueError("This method needs to be overridden.")

  def _build(self, input_shape):
    """Create variables of the Popnn RNN.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    if self.built:
      return

    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.ndims != 3:
      raise ValueError("Expecting input_shape with 3 dims, got %d" %
                       input_shape.ndims)
    input_shape = input_shape.as_list()
    if input_shape[-1] is None:
      raise ValueError("The last dimension of the inputs to `_PopnnRNN` "
                       "should be defined. Found `None`.")
    self._input_size = input_shape[-1]
    self.input_spec = base_layer.InputSpec(ndim=3, axes={-1: self._input_size})

    # Create the variables
    with vs.variable_scope(self._scope, reuse=self.built):
      if self._weights_initializer is None:
        self._weights_initializer = init_ops.glorot_uniform_initializer(
            self._seed, dtype=self._plain_dtype)
      if self._bias_initializer is None:
        self._bias_initializer = init_ops.constant_initializer(
            0.0, dtype=self._plain_dtype)
      self.kernel = vs.get_variable("kernel",
                                    dtype=self._plain_dtype,
                                    initializer=self._weights_initializer,
                                    shape=self.canonical_weight_shape)
      self.biases = vs.get_variable("biases",
                                    dtype=self._plain_dtype,
                                    initializer=self._bias_initializer,
                                    shape=self.canonical_bias_shapes)

    self.built = True

  # pylint: disable=unused-argument
  # pylint: disable=arguments-differ
  def call(self, inputs, initial_state=None, training=True):
    raise ValueError("This method needs to be overridden.")

  def state_shape(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _zero_state(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _canonical_weight_shape(self, layer):
    """Shapes of Popnn canonical weight tensors for given layer."""
    if layer < 0 or layer >= self._num_layers:
      raise ValueError("\'layer\' is not valid, got %s, expecting [%d, %d]" %
                       (layer, 0, self._num_layers - 1))
    if not self._input_size:
      raise RuntimeError(
          "%s._canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    input_size = self._input_size
    num_units = self._num_units
    num_gates = self._num_gates_per_layer

    if layer == 0:
      tf_wts = [input_size, num_units * num_gates]
    else:
      #TODO we only support one layer.
      tf_wts = [num_units, num_units * num_gates]
    tf_wts[0] += num_units
    return tf_wts

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    return [self._num_gates_per_layer, self._num_units]


class PopnnLSTM(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an LSTM layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      lstm = PopnnLSTM(num_units, ...)

      outputs, output_states = lstm(inputs, initial_states, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  @deprecation.deprecated(
      None,
      "The PopnnLSTM layer has been moved to IPU TensorFlow Addons and will be "
      "removed from TensorFlow in a future release.")
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a PopnnLSTM model from model spec.

    Args:
      num_units: the number of units within the LSTM model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weights
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      available_memory_proportion_fwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the forward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used. If available_memory_proportion_bwd is set to None,
        then this value applies to both phases.
      available_memory_proportion_bwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the backward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used.
    """
    super(PopnnLSTM, self).__init__(
        num_units=num_units,
        dtype=dtype,
        partials_dtype=partials_dtype,
        seed=seed,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        recurrent_activation=recurrent_activation,
        name=name,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)

  def build(self, input_shape):
    """Create variables of the PopnnLSTM.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    self._build(input_shape)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the LSTM model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is
        initialized to zeros.
      training: Set to False to use the LSTM model in inference mode.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: An `LSTMStateTuple` of the same shape and structure as
        initial_state.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    batch_size = array_ops.shape(inputs)[1]

    if initial_state is not None and not isinstance(initial_state,
                                                    rnn_cell.LSTMStateTuple):
      raise ValueError("Invalid initial_state type: `%s`, expecting "
                       "`LSTMStateTuple`." % type(initial_state))

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    c, h = initial_state
    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)

    available_memory_proportion_fwd = -1. \
        if self._available_memory_proportion_fwd is None \
        else self._available_memory_proportion_fwd
    available_memory_proportion_bwd = -1. \
        if self._available_memory_proportion_bwd is None \
        else self._available_memory_proportion_bwd

    outputs, output_h, output_c, _ = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    state = rnn_cell.LSTMStateTuple(output_c, output_h)

    return outputs, state

  def state_shape(self, batch_size):
    """Shape of Popnn LSTM states.

    Shape is a 2-element tuple. Each is [batch_size, num_units]

    Args:
      batch_size: an int

    Returns:
      a tuple of Python arrays.
    """
    return ([batch_size, self.num_units], [batch_size, self.num_units])

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(array_ops.zeros(sp, dtype=self.dtype))
    return rnn_cell.LSTMStateTuple(*res)


class PopnnDynamicLSTM(PopnnLSTM):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an LSTM layer,
	  with a sequence length input.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      lstm = PopnnDynamicLSTM(num_units, ...)

      outputs, output_states = lstm(
        inputs, seq_len, initial_state, training=True)

  """
  # pylint:enable=line-too-long

  @deprecation.deprecated(
      None,
      "The PopnnDynamicLSTM layer has been moved to IPU TensorFlow Addons and "
      "will be removed from TensorFlow in a future release.")
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a PopnnDynamicLSTM model from model spec.

    Args:
      num_units: the number of units within the LSTM model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weights
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      available_memory_proportion_fwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the forward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used. If available_memory_proportion_bwd is set to None,
        then this value applies to both phases.
      available_memory_proportion_bwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the backward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used.
    """
    super(PopnnDynamicLSTM, self).__init__(
        num_units=num_units,
        dtype=dtype,
        partials_dtype=partials_dtype,
        seed=seed,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        recurrent_activation=recurrent_activation,
        name=name,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)

  #pylint: disable=W0223
  def call(self, inputs, seq_len, initial_state=None, training=True):
    #pylint: disable=W0221
    """Runs the forward step for the LSTM model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      seq_len: 1-D tensor with the sequence length of samples in each batch.
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is
        initialized to zeros.
      training: Set to False to use the LSTM model in inference mode.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: An `LSTMStateTuple` of the same shape and structure as
        initial_state.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    batch_size = array_ops.shape(inputs)[1]

    if initial_state is not None and not isinstance(initial_state,
                                                    rnn_cell.LSTMStateTuple):
      raise ValueError("Invalid initial_state type: `%s`, expecting "
                       "`LSTMStateTuple`." % type(initial_state))

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    c, h = initial_state
    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)

    available_memory_proportion_fwd = -1. \
        if self._available_memory_proportion_fwd is None \
        else self._available_memory_proportion_fwd
    available_memory_proportion_bwd = -1. \
        if self._available_memory_proportion_bwd is None \
        else self._available_memory_proportion_bwd

    outputs, output_h, output_c, _ = gen_popnn_ops.popnn_dynamic_lstm_layer(
        inputs=inputs,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    state = rnn_cell.LSTMStateTuple(output_c, output_h)

    return outputs, state


class PopnnGRU(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of a GRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gru = PopnnGRU(num_units, ...)

      outputs, output_state = gru(inputs, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_GRU
  _num_gates_per_layer = POPNN_GRU_NUM_GATES

  @deprecation.deprecated(
      None,
      "The PopnnGRU layer has been moved to IPU TensorFlow Addons and will be "
      "removed from TensorFlow in a future release.")
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               reset_after=False,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a PopnnGRU model from model spec.

    Args:
      num_units: the number of units within the GRU model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weights
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      reset_after:  GRU convention (whether to apply reset gate
        after or before matrix multiplication). False = "before" (default),
        True = "after".
        Leave as default (False) to match the behaviour of the standard
        TensorFlow GRU.
      available_memory_proportion_fwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the forward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used. If available_memory_proportion_bwd is set to None,
        then this value applies to both phases.
      available_memory_proportion_bwd: Maximum fraction of IPU memory which can
        be used as temporary scratch space during computation, for the backward
        propagation layer. A value of -1. or None indicates that the default in
        Popnn should be used.
    """
    super(PopnnGRU, self).__init__(
        num_units=num_units,
        dtype=dtype,
        partials_dtype=partials_dtype,
        seed=seed,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        recurrent_activation=recurrent_activation,
        name=name,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    self._reset_after = reset_after

  def build(self, input_shape):
    """Create variables of the PopnnGRU.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    self._build(input_shape)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the GRU model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`. If
        not provided, the state is initialized to zeros.
      training: Set to False to use the GRU model in inference mode.

    Returns:
      A tuple of output and output_state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    batch_size = array_ops.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = ops.convert_to_tensor(initial_state, dtype=dtype)

    available_memory_proportion_fwd = -1. \
        if self._available_memory_proportion_fwd is None \
        else self._available_memory_proportion_fwd
    available_memory_proportion_bwd = -1. \
        if self._available_memory_proportion_bwd is None \
        else self._available_memory_proportion_bwd

    output, output_c, _ = gen_popnn_ops.popnn_gru_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    return output, output_c

  def state_shape(self, batch_size):
    """Shape of Popnn GRU state.

    State shape is [batch_size, num_units].

    Args:
      batch_size: an int

    Returns:
      A Python array.
    """
    return [batch_size, self.num_units]

  def _zero_state(self, batch_size):
    return array_ops.zeros(self.state_shape(batch_size), dtype=self.dtype)

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    if self._reset_after:
      return [self._num_gates_per_layer, 2, self._num_units]
    return super(PopnnGRU, self)._canonical_bias_shape(unused_layer)


class PopnnDynamicGRU(PopnnGRU):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an GRU layer,
  with a sequence length input.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gru = PopnnDynamicGRU(num_units, ...)

      outputs, output_state = gru(
        inputs, seq_len, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_DYNAMIC_GRU
  _num_gates_per_layer = POPNN_DYNAMIC_GRU_NUM_GATES

  @deprecation.deprecated(
      None,
      "The PopnnDynamicGRU layer has been moved to IPU TensorFlow Addons and "
      "will be removed from TensorFlow in a future release.")
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               reset_after=False,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a PopnnDynamicGRU model from model spec.

      Args:
        num_units: the number of units within the RNN model.
        dtype: tf.float16 or tf.float32
        partials_dtype: the type used by Popnn to perform partial calculations.
          Either tf.float16 or tf.float32.
        seed: A Python integer. Used to create the default Glorot uniform
          initializer weights_initializer.
        weights_initializer: starting value to initialize the weight
          (default is Glorot uniform initializer).
        bias_initializer: starting value to initialize the bias
          (default is all zeros).
        activation: Activation function. Defaults to "tanh".
          Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
        recurrent_activation: Recurrent activation function. Defaults to
          "sigmoid". Must generate output in the [0,1] range.
          Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
        name: VariableScope for the created subgraph; defaults to class name.
          This only serves the default scope if later no scope is specified when
          invoking ``__call__()``.
        reset_after:  GRU convention (whether to apply reset gate
          after or before matrix multiplication). False = "before" (default),
          True = "after".
          Leave as default (False) to match the behaviour of the standard
          TensorFlow GRU.
        available_memory_proportion_fwd: Maximum fraction of IPU memory which
          can be used as temporary scratch space during computation, for the
          forward propagation layer. A value of -1. or None indicates that the
          default in Popnn should be used. If available_memory_proportion_bwd is
          set to None, then this value applies to both phases.
        available_memory_proportion_bwd: Maximum fraction of IPU memory which
          can be used as temporary scratch space during computation, for the
          backward propagation layer. A value of -1. or None indicates that the
          default in Popnn should be used.
    """
    super(PopnnDynamicGRU, self).__init__(
        num_units=num_units,
        dtype=dtype,
        partials_dtype=partials_dtype,
        seed=seed,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        recurrent_activation=recurrent_activation,
        name=name,
        reset_after=reset_after,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)

  @property
  def saveable(self):
    return False

  #pylint: disable=arguments-differ
  def call(self,
           inputs,
           seq_len,
           initial_state=None,
           training=True,
           time_major=True):
    """Runs the forward step for the DynamicGRU model.

      Args:
        inputs: 3-D tensor with shape [batch_size, time_len, input_size].
        seq_len: 1-D tensor with the sequence length of samples in each batch.
        initial_state: Initial state tensor, shaped `[batch_size, num_units]`.
          If not provided, the state is initialized to zeros.
        training: whether this operation will be used in training or inference.
        time_major: whether the time dimension is the first demension.

      Returns:
        A tuple of output and output state.

        * output: a tensor of shape [time_len, batch_size, num_units].
        * output_state: The output state of the last cell.

      Raises:
        ValueError: if initial_state is not valid.

    """

    dtype = self.dtype

    inputs = ops.convert_to_tensor(inputs, dtype=dtype)
    if not time_major:
      inputs = array_ops.transpose(inputs, [1, 0, 2])

    batch_size = array_ops.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = ops.convert_to_tensor(initial_state, dtype=dtype)
    bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
    bias_zeros = init_ops.constant_initializer(0.0, dtype=inputs.dtype)
    biases_r_u = vs.get_variable("bias_r_u",
                                 dtype=inputs.dtype,
                                 initializer=bias_ones,
                                 shape=[2, self._num_units])
    biases_c = vs.get_variable("bias_c",
                               dtype=inputs.dtype,
                               initializer=bias_zeros,
                               shape=[1, self._num_units])
    biases = array_ops.concat([biases_r_u, biases_c], axis=0)
    if self._reset_after:
      biases = array_ops.expand_dims(biases, 1)
      biases = array_ops.concat([biases, biases], axis=1)
    self.biases = biases

    available_memory_proportion_fwd = -1. \
        if self._available_memory_proportion_fwd is None \
        else self._available_memory_proportion_fwd
    available_memory_proportion_bwd = -1. \
        if self._available_memory_proportion_bwd is None \
        else self._available_memory_proportion_bwd

    output, output_c, _ = gen_popnn_ops.popnn_dynamic_gru_layer(
        inputs=inputs,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    return output, output_c


class PopnnAUGRU(PopnnGRU):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an AUGRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      augru = PopnnAUGRU(num_units, ...)

      outputs, output_state = augru(inputs, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_AUGRU
  _num_gates_per_layer = POPNN_AUGRU_NUM_GATES

  @deprecation.deprecated(
      None, "The PopnnAUGRU layer has been moved to IPU TensorFlow Addons and "
      "will be removed from TensorFlow in a future release.")
  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               name=None,
               reset_after=False,
               available_memory_proportion_fwd=None,
               available_memory_proportion_bwd=None):
    """Creates a PopnnAUGRU model from model spec.

    Args:
      num_units: the number of units within the RNN model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weight
        (default is Glorot uniform initializer).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      available_memory_proportion_fwd: Maximum fraction of IPU memory which
        can be used as temporary scratch space during computation, for the
        forward propagation layer. A value of -1. or None indicates that the
        default in Popnn should be used. If available_memory_proportion_bwd is
        set to None, then this value applies to both phases.
      available_memory_proportion_bwd: Maximum fraction of IPU memory which
        can be used as temporary scratch space during computation, for the
        backward propagation layer. A value of -1. or None indicates that the
        default in Popnn should be used.
    """
    super(PopnnAUGRU, self).__init__(
        num_units=num_units,
        dtype=dtype,
        partials_dtype=partials_dtype,
        seed=seed,
        weights_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        activation=activation,
        recurrent_activation=recurrent_activation,
        name=name,
        reset_after=reset_after,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)

  #pylint: disable=arguments-differ
  def call(self,
           inputs,
           seq_len,
           attention_score,
           initial_state=None,
           training=True,
           time_major=True):
    """Runs the forward step for the AUGRU model.

    Args:
      inputs: 3-D tensor with shape [time_len, batch_size, input_size].
      seq_len: 1-D tensor with the sequence length of samples in each batch.
      attention_score: The output of attention layer, the score of samples
        in each batch, shaped `[batch_size, max_seq_len]`.
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`.
        If not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.
      time_major: whether the time dimension is the first dimension.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)
    if not time_major:
      inputs = array_ops.transpose(inputs, [1, 0, 2])
      attention_score = array_ops.transpose(attention_score, [1, 0])

    batch_size = array_ops.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = ops.convert_to_tensor(initial_state, dtype=dtype)
    augru_biases_r_u = vs.get_variable("bias_r_u",
                                       dtype=inputs.dtype,
                                       initializer=init_ops.ones_initializer(),
                                       shape=[2, self._num_units])
    augru_biases_c = vs.get_variable("bias_c",
                                     dtype=inputs.dtype,
                                     initializer=init_ops.zeros_initializer(),
                                     shape=[1, self._num_units])
    augru_biases = array_ops.concat([augru_biases_r_u, augru_biases_c], axis=0)
    if self._reset_after:
      augru_biases = array_ops.expand_dims(augru_biases, 1)
      augru_biases = array_ops.concat([augru_biases, augru_biases], axis=1)

    available_memory_proportion_fwd = -1. \
        if self._available_memory_proportion_fwd is None \
        else self._available_memory_proportion_fwd
    available_memory_proportion_bwd = -1. \
        if self._available_memory_proportion_bwd is None \
        else self._available_memory_proportion_bwd

    output, output_c, _ = gen_popnn_ops.popnn_augru_layer(
        inputs=inputs,
        att_score=attention_score,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=augru_biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        available_memory_proportion_fwd=available_memory_proportion_fwd,
        available_memory_proportion_bwd=available_memory_proportion_bwd)
    return output, output_c

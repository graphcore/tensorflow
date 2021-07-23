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
Recurrent Keras layers
~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ipu import rand_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.util import deprecation

from tensorflow.python.ops import rnn_cell
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops

POPNN_LSTM = "lstm"
POPNN_GRU = "gru"

POPNN_LSTM_NUM_GATES = 4
POPNN_GRU_NUM_GATES = 3

__all__ = ["PopnnLSTM", "PopnnGRU"]

ACCEPTED_ACTIVATIONS = ['tanh', 'relu', 'softmax', 'sigmoid', 'hard_sigmoid']
ACCEPTED_RECURRENT_ACTIVATIONS = ['tanh', 'softmax', 'sigmoid', 'hard_sigmoid']

ACCEPTED_ACTIVATIONS_STR = ",".join(
    map(lambda s: f'"{s}"', ACCEPTED_ACTIVATIONS))
ACCEPTED_RECURRENT_ACTIVATIONS_STR = ",".join(
    map(lambda s: f'"{s}"', ACCEPTED_RECURRENT_ACTIVATIONS))


class _PopnnRNN(Layer):
  """Base class for implementing XLA and Popnn compatible RNN layers.
  """
  def __init__(self,
               num_units,
               activation='tanh',
               recurrent_activation='sigmoid',
               partials_dtype=dtypes.float32,
               seed=None,
               dropout_seed=None,
               kernel_initializer=None,
               recurrent_initializer=None,
               bias_initializer=None,
               dtype=dtypes.float32,
               dropout=0.,
               return_state=False,
               return_sequences=False,
               time_major=False,
               stateful=False,
               **kwargs):
    super(_PopnnRNN, self).__init__(dtype=dtype, **kwargs)

    activation = op_util.get_activation_name(activation)
    recurrent_activation = op_util.get_activation_name(recurrent_activation)

    if activation not in ACCEPTED_ACTIVATIONS:
      raise ValueError("IPU custom RNN layer does not support '" + activation +
                       "' as an activation. Acceptable value are " +
                       ACCEPTED_ACTIVATIONS_STR)

    if recurrent_activation not in ACCEPTED_RECURRENT_ACTIVATIONS:
      raise ValueError("IPU custom RNN layer does not support '" +
                       recurrent_activation + "' as a "
                       "recurrent activation. Acceptable value are " +
                       ACCEPTED_RECURRENT_ACTIVATIONS_STR)

    if dtype not in [dtypes.float16, dtypes.float32]:
      raise ValueError("Only support float16, float32, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._activation = activation
    self._recurrent_activation = recurrent_activation
    self._plain_dtype = dtype
    self._partials_dtype = partials_dtype
    self._num_units = num_units
    self._kernel_initializer = kernel_initializer
    self._recurrent_initializer = recurrent_initializer
    self._bias_initializer = bias_initializer
    self._dropout = dropout
    self._dropout_seed = dropout_seed
    self._seed = seed
    self._return_state = return_state
    self._return_sequences = return_sequences
    self._time_major = time_major
    self._stateful = stateful
    # Initialize input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

  def _verify_input_size(self):
    if not self._input_size:
      raise ValueError(
          "`_input_size` is unknown because the layer has not been built yet.")

  @property
  def num_units(self):
    return self._num_units

  @property
  def input_size(self):
    self._verify_input_size()
    return self._input_size

  @property
  def canonical_weight_shape(self):
    """Shapes of Popnn canonical weight tensors."""
    self._verify_input_size()
    return [
        self._input_size + self._num_units,
        self._num_units * self._num_gates_per_layer
    ]

  @property
  def canonical_bias_shapes(self):
    """Shapes of Popnn canonical bias tensors."""
    self._verify_input_size()
    return self._canonical_bias_shape(0)

  # pylint: disable=unused-argument, arguments-differ
  def build(self, input_shape):
    raise ValueError("This method needs to be overridden.")

  def _build(self, input_shape):
    """Create variables of the Popnn RNN layer.

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

    # Create the variables
    if self._kernel_initializer is None:
      self._kernel_initializer = init_ops.glorot_uniform_initializer(
          self._seed)

    if self._recurrent_initializer is None:
      self._recurrent_initializer = init_ops.orthogonal_initializer(self._seed)

    if self._bias_initializer is None:
      self._bias_initializer = init_ops.zeros_initializer()

    self._kernel_initializer = initializers.get(self._kernel_initializer)
    self._recurrent_initializer = initializers.get(self._recurrent_initializer)
    self._bias_initializer = initializers.get(self._bias_initializer)

    # Initialize the input weight tensor.
    kernel_shape = self.canonical_weight_shape
    kernel_shape[0] -= self.num_units
    self.kernel = self.add_weight("kernel",
                                  dtype=self._plain_dtype,
                                  initializer=self._kernel_initializer,
                                  shape=kernel_shape)

    # Initialize the recurrent weight tensor.
    recurrent_kernel_shape = self.canonical_weight_shape
    recurrent_kernel_shape[0] = self.num_units
    self.recurrent_kernel = self.add_weight(
        "recurrent_kernel",
        dtype=self._plain_dtype,
        initializer=self._recurrent_initializer,
        shape=recurrent_kernel_shape)

    self.biases = self.get_bias()

    self.states = []
    if self._stateful:
      batch_size = input_shape[1 if self._time_major else 0]
      shapes = self.state_shape(batch_size)
      if not isinstance(shapes, tuple):
        shapes = (shapes,)

      for i, shape in enumerate(shapes):
        self.states.append(K.zeros(shape))

    self.built = True

  def get_bias(self):
    return self.add_weight("biases",
                           dtype=self._plain_dtype,
                           initializer=self._bias_initializer,
                           shape=self.canonical_bias_shapes)

  # pylint: disable=unused-argument
  def call(self, inputs, training=None, initial_state=None):
    raise ValueError("This method needs to be overridden.")

  # pylint: disable=unused-argument
  def state_shape(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  # pylint: disable=unused-argument
  def _zero_state(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    return [self._num_gates_per_layer * self._num_units]

  def _apply_dropout(self, inputs, training):
    if not training:
      return inputs

    # Apply the same dropout mask across the sequence - this function is called
    # when the inputs is shaped as [S, B, N].
    noise_shape = inputs.get_shape().as_list()
    noise_shape[0] = 1

    return rand_ops.dropout(inputs,
                            seed=self._dropout_seed,
                            rate=self._dropout,
                            noise_shape=noise_shape,
                            name=self.name + "_dropout")


class PopnnLSTM(_PopnnRNN):
  # pylint:disable=line-too-long
  """Popnn implementation of Long Short-Term Memory layer (Hochreiter and
  Schmidhuber 1997), optimized for the IPU.

  Note that the Keras equivalent uses the `hard_sigmoid` as the default
  recurrent activation, however this version uses `sigmoid` as the default.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent ("tanh").
      Accepted activations: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid ("sigmoid").
      Accepted activations: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean. If True then the layer will use a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True then add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in `Jozefowicz et al
      <http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf>`_.
    kernel_regularizer: Unsupported - Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Unsupported - Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Unsupported - Regularizer function applied to the bias
      vector.
    activity_regularizer: Unsupported - Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Unsupported - Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Unsupported - Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Unsupported - Constraint function applied to the bias
      vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    dropout_seed: An optional two-element tensor-like object (`tf.Tensor`, a
      numpy array or Python list/tuple), representing the random seed that will
      be used to create the distribution for dropout.
    recurrent_dropout: Unsupported - Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Unsupported - Implementation mode.
    return_sequences: Boolean. If True then the full output sequence will be
      returned.
      If False then only the last output in the output sequence will be
      returned.
    return_state: Boolean. If True then the last state will be returned
      in addition to the last output or output sequence.
    go_backwards: Unsupported - Boolean (default False).
      If True process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Unsupported - Boolean (default False).
      If True the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    seed: A Python integer. Used for the `kernel_initializer` and
      `recurrent_initializer`.
    partials_dtype: the type used by Popnn to perform partial calculations.
      Either tf.float16 or tf.float32.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True the shape of the inputs and outputs will be
      `(timesteps, batch, ...)`, otherwise the shape will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  @deprecation.deprecated(
      None,
      "Please move your model to TensorFlow 2 which has full Keras support for "
      "IPU.")
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               dropout_seed=None,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               partials_dtype=dtypes.float32,
               seed=None,
               time_major=False,
               **kwargs):

    if not use_bias:
      raise ValueError(
          "IPU custom LSTM layer does not support use_bias = False.")

    if kernel_regularizer:
      raise ValueError(
          "IPU custom LSTM layer does not support kernel_regularizer.")

    if recurrent_regularizer:
      raise ValueError(
          "IPU custom LSTM layer does not support recurrent_regularizer.")

    if bias_regularizer:
      raise ValueError(
          "IPU custom LSTM layer does not support bias_regularizer.")

    if activity_regularizer:
      raise ValueError(
          "IPU custom LSTM layer does not support activity_regularizer.")

    if kernel_constraint:
      raise ValueError(
          "IPU custom LSTM layer does not support kernel_constraint.")

    if recurrent_constraint:
      raise ValueError(
          "IPU custom LSTM layer does not support recurrent_constraint.")

    if bias_constraint:
      raise ValueError(
          "IPU custom LSTM layer does not support bias_constraint.")

    if recurrent_dropout != 0.:
      raise ValueError(
          "IPU custom LSTM layer does not support recurrent_dropout.")

    if implementation != 1:
      raise ValueError(
          "IPU custom LSTM layer does not support implementation != 1.")

    if go_backwards:
      raise ValueError(
          "IPU custom LSTM layer does not support go_backwards = True.")

    if unroll:
      raise ValueError("IPU custom LSTM layer does not support unroll = True.")

    super(PopnnLSTM,
          self).__init__(num_units=units,
                         activation=activation,
                         recurrent_activation=recurrent_activation,
                         partials_dtype=partials_dtype,
                         seed=seed,
                         kernel_initializer=kernel_initializer,
                         recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer,
                         dropout=dropout,
                         dropout_seed=dropout_seed,
                         return_state=return_state,
                         return_sequences=return_sequences,
                         time_major=time_major,
                         stateful=stateful,
                         **kwargs)
    self.unit_forget_bias = unit_forget_bias

  def build(self, input_shape):
    """Create variables of the PopnnLSTM layer.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd
                  dimension.
    """
    self._build(input_shape)

  def get_bias(self):
    if self.unit_forget_bias:

      def bias_initializer(_, *args, **kwargs):
        # Forget gate is the second slice.
        init = K.concatenate([
            self._bias_initializer((1, self.num_units), *args, **kwargs),
            initializers.Ones()((1, self.num_units), *args, **kwargs),
            self._bias_initializer((2, self.num_units), *args, **kwargs),
        ],
                             axis=0)
        return array_ops.reshape(init, self.canonical_bias_shapes)
    else:
      bias_initializer = self._bias_initializer

    return self.add_weight("biases",
                           dtype=self._plain_dtype,
                           initializer=bias_initializer,
                           shape=self.canonical_bias_shapes)

  def call(self, inputs, training=None, initial_state=None):
    """Runs the forward step for the LSTM layer.

    Args:
      inputs: 3D tensor with shape [batch_size, seq_len, input_size]. If the
              time_major parameter is set to True then the shape should
              be [seq_len, batch_size, input_size].
      training: Set to False to use the layer in inference mode. This is only
        relevant if `dropout` or `recurrent_dropout` is set.
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is initialized to
        zeros.

    Returns:
      If `return_sequences` is True the LSTM layer returns a tensor of
      shape [batch_size, seq_len, num_units] otherwise it returns
      a tensor of shape [batch_size, num_units].
      If `return_state` is True then the output state of the last cell is also
      returned.

    """
    if training is None:
      training = K.learning_phase()

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    if len(inputs.shape) != 3:
      raise ValueError("inputs tensor must be 3D")

    if not self._time_major:
      # Shuffle from Keras [B, S, N] to PopLibs [S, B, N]
      inputs = array_ops.transpose(inputs, [1, 0, 2])

    batch_size = array_ops.shape(inputs)[1]

    # PopnnLSTM doesn't support a dynamic training parameter. If the training
    # parameter is not constant, assume training.
    training = training if isinstance(training, bool) else True

    if initial_state is not None:
      pass
    elif self._stateful:
      initial_state = (self.states[0], self.states[1])
    else:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    combined_kernel = array_ops.concat([self.kernel, self.recurrent_kernel], 0)

    h, c = initial_state

    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)

    if self._dropout > 0.:
      inputs = self._apply_dropout(inputs, training)

    bias_tensor = array_ops.reshape(
        self.biases, [self._num_gates_per_layer, self._num_units])

    output, output_h, output_c, _ = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        num_channels=self._num_units,
        kernel=combined_kernel,
        biases=bias_tensor,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)

    if self._stateful:
      updates = []
      for state_, state in zip(self.states, (output_h, output_c)):
        updates.append(state_ops.assign(state_, state))
      self.add_update(updates)

    if not self._time_major:
      # Convert output from PopLibs [S, B, N] to Keras [B, S, N]
      output = array_ops.transpose(output, [1, 0, 2])

    if not self._return_sequences:
      output = output[-1, :, :] if self._time_major else output[:, -1, :]

    if self._return_state:
      return output, output_h, output_c

    return output

  def state_shape(self, batch_size):
    """Shape of Popnn LSTM states.

    Shape is a 2-element tuple. Each is [batch_size, num_units]

    Args:
      batch_size: an int

    Returns:
      A tuple of Python arrays.
    """
    return ([batch_size, self.num_units], [batch_size, self.num_units])

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(array_ops.zeros(sp, dtype=self.dtype))
    return rnn_cell.LSTMStateTuple(*res)


class PopnnGRU(_PopnnRNN):
  # pylint:disable=line-too-long
  """Popnn implementation of the Gated Recurrent Unit (Cho et al. 2014),
  optimized for the IPU.

  There are two variants of the GRU implementation. The default is based on
  `v3 <https://arxiv.org/abs/1406.1078v3>`_ and has reset gate applied to hidden
  state before matrix multiplication. The other is based on the
  `original version <https://arxiv.org/abs/1406.1078v1>`_ and has the order
  reversed.
  The first one is the default behaviour for this implementation, however the
  Keras equivalent can use the second variant. To use this variant,
  set `'reset_after'=True`.

  Note that the Keras equivalent uses the `hard_sigmoid` as the default
  recurrent activation, however this version uses `sigmoid` as the default.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent ("tanh").
      Accepted activations: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid ("sigmoid").
      Accepted activations: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean. If True then the layer will use a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer:  Unsupported - Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer:  Unsupported - Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer:  Unsupported - Regularizer function applied to the bias
      vector.
    activity_regularizer:  Unsupported - Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint:  Unsupported - Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint:  Unsupported - Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint:  Unsupported - Constraint function applied to the bias
      vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    dropout_seed: An optional two-element tensor-like object (`tf.Tensor`, a
      numpy array or Python list/tuple), representing the random seed that will
      be used to create the distribution for dropout.
    recurrent_dropout:  Unsupported - Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation:  Unsupported - Implementation mode.
    return_sequences: Boolean. If True then the full output sequence will be
      returned.
      If False then only the last output in the output sequence will be
      returned.
    return_state: Boolean. If True then the last state will be returned
      in addition to the last output or output sequence.
    go_backwards:  Unsupported - Boolean (default False).
      If True process the input sequence backwards and return the
      reversed sequence.
    stateful:  Boolean (default False). If True the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll:  Unsupported - Boolean (default False).
      If True the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True the shape of the inputs and outputs will be
      `(timesteps, batch, ...)`, otherwise the shape will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    seed: A Python integer. Used for the `kernel_initializer` and
      `recurrent_initializer`.
    partials_dtype: the type used by Popnn to perform partial calculations.
      Either tf.float16 or tf.float32.
    reset_after:  GRU convention (whether to apply reset gate
      after or before matrix multiplication). False = "before",
      True = "after" (default).
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_GRU
  _num_gates_per_layer = POPNN_GRU_NUM_GATES

  @deprecation.deprecated(
      None,
      "Please move your model to TensorFlow 2 which has full Keras support for "
      "IPU.")
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               dropout_seed=None,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=True,
               seed=None,
               partials_dtype=dtypes.float32,
               time_major=False,
               **kwargs):

    if not use_bias:
      raise ValueError(
          "IPU custom GRU layer does not support use_bias = False.")

    if kernel_regularizer:
      raise ValueError(
          "IPU custom GRU layer does not support kernel_regularizer.")

    if recurrent_regularizer:
      raise ValueError(
          "IPU custom GRU layer does not support recurrent_regularizer.")

    if bias_regularizer:
      raise ValueError(
          "IPU custom GRU layer does not support bias_regularizer.")

    if activity_regularizer:
      raise ValueError(
          "IPU custom GRU layer does not support activity_regularizer.")

    if kernel_constraint:
      raise ValueError(
          "IPU custom GRU layer does not support kernel_constraint.")

    if recurrent_constraint:
      raise ValueError(
          "IPU custom GRU layer does not support recurrent_constraint.")

    if bias_constraint:
      raise ValueError(
          "IPU custom GRU layer does not support bias_constraint.")

    if recurrent_dropout != 0.:
      raise ValueError(
          "IPU custom GRU layer does not support recurrent_dropout != 0.")

    if implementation != 1:
      raise ValueError(
          "IPU custom GRU layer does not support implementation != 1.")

    if go_backwards:
      raise ValueError("IPU custom GRU layer does not support go_backwards.")

    if unroll:
      raise ValueError("IPU custom GRU layer does not support unroll.")

    self._reset_after = reset_after

    super(PopnnGRU, self).__init__(num_units=units,
                                   activation=activation,
                                   recurrent_activation=recurrent_activation,
                                   partials_dtype=partials_dtype,
                                   seed=seed,
                                   kernel_initializer=kernel_initializer,
                                   recurrent_initializer=recurrent_initializer,
                                   bias_initializer=bias_initializer,
                                   dropout=dropout,
                                   dropout_seed=dropout_seed,
                                   return_state=return_state,
                                   return_sequences=return_sequences,
                                   time_major=time_major,
                                   stateful=stateful,
                                   **kwargs)

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    if self._reset_after:
      return [2, self._num_gates_per_layer * self._num_units]
    return super()._canonical_bias_shape(unused_layer)

  def build(self, input_shape):
    """Create variables of the PopnnGRU layer.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd
                  dimension.
    """
    self._build(input_shape)

  def call(self, inputs, training=None, initial_state=None):
    """Runs the forward step for the GRU layer.

    Args:
      inputs: 3D tensor with shape [batch_size, seq_len, input_size]. If the
              time_major parameter is True, the the shape should be
              [seq_len, batch_size, input_size].
      training: Set to False to use the layer in inference mode.
        This is only relevant if `dropout` or `recurrent_dropout` is used.
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`
        If not provided, the state is initialized to zeros.

    Returns:
      If `return_sequences` is True then the GRU layer returns a tensor of
      shape [batch_size, seq_len, num_units], otherwise it returns
      a tensor of shape [batch_size, num_units].
      If `return_state` is set to True then the output state of the last cell
      is also returned.

    Raises:
      ValueError: if initial_state is not valid.

    """
    if training is None:
      training = K.learning_phase()

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    if len(inputs.shape) != 3:
      raise ValueError("inputs tensor must be 3D")

    if not self._time_major:
      # Shuffle from Keras [B, S, N] to PopLibs [S, B, N]
      inputs = array_ops.transpose(inputs, [1, 0, 2])

    batch_size = array_ops.shape(inputs)[1]

    # PopnnGRU doesn't support a dynamic training parameter. If the training
    # parameter is not constant, assume training.
    training = training if isinstance(training, bool) else True

    if initial_state is not None:
      pass
    elif self._stateful:
      initial_state = self.states[0]
    else:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = ops.convert_to_tensor(initial_state, dtype=dtype)

    if self._dropout > 0.:
      inputs = self._apply_dropout(inputs, training)

    combined_kernel = array_ops.concat([self.kernel, self.recurrent_kernel], 0)

    if self._reset_after:
      # New shape: [self._num_gates_per_layer, 2, self._num_units]
      bias_tensor = array_ops.stack(
          array_ops.split(self.biases,
                          [self._num_units] * self._num_gates_per_layer,
                          axis=1))
    else:
      bias_tensor = array_ops.reshape(
          self.biases, [self._num_gates_per_layer, self._num_units])

    output, output_state, _ = gen_popnn_ops.popnn_gru_layer(
        inputs=inputs,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        num_channels=self._num_units,
        kernel=combined_kernel,
        biases=bias_tensor,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name,
        reset_after=self._reset_after)

    if self._stateful:
      updates = [state_ops.assign(self.states[0], output_state)]
      self.add_update(updates)

    if not self._time_major:
      # Convert output from PopLibs [S, B, N] to Keras [B, S, N]
      output = array_ops.transpose(output, [1, 0, 2])

    if not self._return_sequences:
      output = output[-1, :, :] if self._time_major else output[:, -1, :]

    if self._return_state:
      return output, output_state

    return output

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


# Alias the 2 classes without the Popnn prefix
LSTM = PopnnLSTM
GRU = PopnnGRU

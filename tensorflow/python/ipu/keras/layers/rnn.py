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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ipu import rand_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

from tensorflow.python.ops import rnn_cell
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops

from tensorflow.python.keras import initializers

POPNN_LSTM = "lstm"
POPNN_GRU = "gru"

POPNN_LSTM_NUM_GATES = 4
POPNN_GRU_NUM_GATES = 3

__all__ = ["PopnnLSTM", "PopnnGRU"]


class _PopnnRNN(Layer):
  """Base class for implementing XLA and Popnn compatible RNN layers.
  """
  def __init__(self,
               num_units,
               partials_dtype=dtypes.float32,
               seed=None,
               dropout_seed=None,
               kernel_initializer=None,
               bias_initializer=None,
               dtype=dtypes.float32,
               dropout=0.,
               return_state=False,
               return_sequences=False,
               time_major=False,
               **kwargs):
    super(_PopnnRNN, self).__init__(dtype=dtype, **kwargs)

    if dtype not in [dtypes.float16, dtypes.float32]:
      raise ValueError("Only support float16, float32, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._plain_dtype = dtype
    self._partials_dtype = partials_dtype
    self._num_units = num_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._dropout = dropout
    self._dropout_seed = dropout_seed
    self._seed = seed
    self._return_state = return_state
    self._return_sequences = return_sequences
    self._time_major = time_major
    # Init input_size to None, which will be set after build().
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

  def _build(self, input_shape, recurrent_weight_init=None):
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

    # Create the variables
    if self._kernel_initializer is None:
      self._kernel_initializer = init_ops.glorot_uniform_initializer(
          self._seed, dtype=self._plain_dtype)
    if self._bias_initializer is None:
      self._bias_initializer = init_ops.constant_initializer(
          0.0, dtype=self._plain_dtype)

    # The normal Keras LSTM layers have two sets of weights one for the
    # input and one for the previous recurrent state. By default we have
    # just one big weight with both of them. This is functionally the same
    # BUT it means that the initalizer will be applied differently so we
    # need to initalize them seperately then concat them so the init
    # behaviour is the same as keras.
    if recurrent_weight_init is None:
      self.kernel = self.add_weight("kernel",
                                    dtype=self._plain_dtype,
                                    initializer=self._kernel_initializer,
                                    shape=self.canonical_weight_shape)
      self.recurrent_kernel = None
    else:
      # Initialize the input weight tensor.
      input_kernel_shape = self.canonical_weight_shape
      input_kernel_shape[0] -= self.num_units
      self.kernel = self.add_weight("kernel",
                                    dtype=self._plain_dtype,
                                    initializer=self._kernel_initializer,
                                    shape=input_kernel_shape)

      # Initialize the recurrent weight tensor.
      recurrent_kernel_shape = self.canonical_weight_shape
      recurrent_kernel_shape[0] = self.num_units
      self.recurrent_kernel = self.add_weight(
          "recurrent_kernel",
          dtype=self._plain_dtype,
          initializer=recurrent_weight_init,
          shape=recurrent_kernel_shape)

    self.biases = self.add_weight("biases",
                                  dtype=self._plain_dtype,
                                  initializer=self._bias_initializer,
                                  shape=self.canonical_bias_shapes)

    self.built = True

  # pylint: disable=unused-argument
  def call(self, inputs, initial_state=None, training=True):
    raise ValueError("This method needs to be overridden.")

  # pylint: disable=unused-argument
  def state_shape(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  # pylint: disable=unused-argument
  def _zero_state(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    return [self._num_gates_per_layer, self._num_units]

  def _apply_dropout(self, inputs, training):
    if not training:
      return inputs

    if self._dropout_seed is None:
      # User did not provide a seed
      self._dropout_seed = [0, 0]

    return rand_ops.dropout(inputs,
                            seed=self._dropout_seed,
                            rate=self._dropout,
                            scale=1.,
                            name=self.name + "_dropout")


class PopnnLSTM(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, Popnn implementation of an LSTM layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      lstm = PopnnLSTM(num_units, ...)

      outputs = lstm(inputs)

  Args:
    num_units: the number of units within the RNN model.
    partials_dtype: the type used by Popnn to perform partial calculations.
      Either tf.float16 or tf.float32.
    kernel_initializer: starting value to initialize the weight (default is all
      zeros).
    bias_initializer: starting value to initialize the bias (default is all
      zeros).
    recurrent_initializer: This optional parameter will partition weight
      initialization into two stages, first initalizing the input kernel using
      kernel_initializer then will initalize a kernel for the recurrent state.
      This partitioning is what the keras LSTM layer does (default is None,
      meaning off).
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs.
    dropout_seed: An optional two-element tensor-like object (`tf.Tensor`, a
      numpy array or Python list/tuple), representing the random seed that will
      be used to create the distribution for dropout.
    return_state: When True, the layer returns a tuple containing the
      output and the state tensors.  Otherwise it returns only the
      output tensor.
    seed: A Python integer. Used to create the default Glorot uniform
      initializer kernel_initializer.
    time_major: The input should be of the form [sequence, batch, units]
                instead of the default [batch, sequence, units].
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
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
    if recurrent_initializer is not None:
      self.recurrent_initializer = initializers.get(recurrent_initializer)
    else:
      self.recurrent_initializer = None

    if activation != 'tanh':
      raise ValueError(
          "IPU custom LSTM layer does not support alternative activations.")

    if recurrent_activation != 'hard_sigmoid':
      raise ValueError(
          "IPU custom LSTM layer does not support alternative recurrent "
          "activations.")

    if not use_bias:
      raise ValueError(
          "IPU custom LSTM layer does not support use_bias = False.")

    if not unit_forget_bias:
      raise ValueError(
          "IPU custom LSTM layer does not support unit_forget_bias = False.")

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

    if stateful:
      raise ValueError(
          "IPU custom LSTM layer does not support stateful = True.")

    if unroll:
      raise ValueError("IPU custom LSTM layer does not support unroll = True.")

    super(PopnnLSTM, self).__init__(num_units=units,
                                    partials_dtype=partials_dtype,
                                    seed=seed,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    dropout=dropout,
                                    dropout_seed=dropout_seed,
                                    return_state=return_state,
                                    return_sequences=return_sequences,
                                    time_major=time_major,
                                    **kwargs)

  def build(self, input_shape):
    """Create variables of the PopnnLSTM.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd
                  dimension.
    """
    self._build(input_shape, self.recurrent_initializer)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the LSTM layer.

    Args:
      inputs: 3-D tensor with shape [batch_size, seq_len, input_size]. If the
              time_major parameter is set to True, then the shape should
              be [seq_len, batch_size, input_size].
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is initialized to
        zeros.
      training: whether this operation will be used in training or inference.

    Returns:
      output: When `return_sequences` is set, then LSTM returns a tensor of
              shape [batch_size, seq_len, num_units], otherwise it returns
              a tensor of shape [batch_size, num_units].
      output_state: The output state of the last cell, when the parameter
                    `return_state` is set to True.

    """
    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    if len(inputs.shape) != 3:
      raise ValueError("inputs tensor must be 3D")

    if not self._time_major:
      # Shuffle from Keras [B, S, N] to Poplibs [S, B, N]
      inputs = array_ops.transpose(inputs, [1, 0, 2])

    batch_size = array_ops.shape(inputs)[1]

    # PopnnLSTM doesn't support a dynamic training parameter.
    if not isinstance(training, bool):
      raise ValueError(
          "PopnnLSTM does not support a dynamic training argument.  Please "
          "pass a boolean True/False to the call method.  If you are using "
          "keras.Sequential, you should change to another model type.")

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    # If a recurrent kernel (that is, a seperate kernel which applies to
    # the recurrent state with the normal kernel applying to the inputs)
    # was provided then we combine it to create the full kernel.
    if self.recurrent_kernel is not None:
      combined_kernel = array_ops.concat([self.kernel, self.recurrent_kernel],
                                         0)
    else:
      combined_kernel = self.kernel

    c, h = initial_state

    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)

    if self._dropout > 0.:
      inputs = self._apply_dropout(inputs, training)

    output, output_h, output_c, _ = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=combined_kernel,
        biases=self.biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)

    if not self._time_major:
      # Convert output from Poplibs [S, B, N] to Keras [B, S, N]
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
      a tuple of python arrays.
    """
    return ([batch_size, self.num_units], [batch_size, self.num_units])

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(array_ops.zeros(sp, dtype=self.dtype))
    return rnn_cell.LSTMStateTuple(*res)


class PopnnGRU(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, Popnn implementation of an GRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gru = PopnnGRU(num_units, ...)

      outputs = gru(inputs)

  Args:
    units: the number of units within the RNN model.
    partials_dtype: the type used by Popnn to perform partial calculations.
      Either tf.float16 or tf.float32.
    kernel_initializer: starting value to initialize the weight (default is
      Glorot uniform initializer).
    bias_initializer: starting value to initialize the bias (default is all
      zeros).
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs.
    dropout_seed: An optional two-element tensor-like object (`tf.Tensor`, a
      numpy array or Python list/tuple), representing the random seed that will
      be used to create the distribution for dropout.
    return_state: When True, the layer returns a tuple containing the output and
      the state tensors.  Otherwise it returns only the output tensor.
    seed: A Python integer. Used to create the default Glorot uniform
      initializer kernel_initializer.
    time_major: The input should be of the form [sequence, batch, units]
                instead of the default [batch, sequence, units].
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_GRU
  _num_gates_per_layer = POPNN_GRU_NUM_GATES

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
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
               reset_after=False,
               partials_dtype=dtypes.float32,
               seed=None,
               time_major=False,
               **kwargs):

    if activation != 'tanh':
      raise ValueError("IPU custom GRU layer does not support activation.")

    if recurrent_activation != 'hard_sigmoid':
      raise ValueError(
          "IPU custom GRU layer does not support recurrent_activation.")

    if not use_bias:
      raise ValueError(
          "IPU custom GRU layer does not support use_bias = True.")

    if recurrent_initializer != 'orthogonal':
      raise ValueError(
          "IPU custom GRU layer does not support recurrent_initializer.")

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

    if stateful:
      raise ValueError("IPU custom GRU layer does not support stateful.")

    if unroll:
      raise ValueError("IPU custom GRU layer does not support unroll.")

    if reset_after:
      raise ValueError("IPU custom GRU layer does not support reset_after.")

    super(PopnnGRU, self).__init__(num_units=units,
                                   partials_dtype=partials_dtype,
                                   seed=seed,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   dropout=dropout,
                                   dropout_seed=dropout_seed,
                                   return_state=return_state,
                                   return_sequences=return_sequences,
                                   time_major=time_major,
                                   **kwargs)

  def build(self, input_shape):
    """Create variables of the PopnnGRU.

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

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the GRU layer.

    Args:
      inputs: 3-D tensor with shape [batch_size, seq_len, input_size]. If the
              time_major parameter is True, the the shape should be
              [seq_len, batch_size, input_size].
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`
        If not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.

    Returns:
      output: When `return_sequences` is set, then GRU returns a tensor of
              shape [batch_size, seq_len, num_units], otherwise it returns
              a tensor of shape [batch_size, num_units].
      output_state: The output state of the last cell, when the parameter
                    `return_state` is set to True.

    Raises:
      ValueError: if initial_state is not valid.

    """
    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    if len(inputs.shape) != 3:
      raise ValueError("inputs tensor must be 3D")

    if not self._time_major:
      # Shuffle from Keras [B, S, N] to Poplibs [S, B, N]
      inputs = array_ops.transpose(inputs, [1, 0, 2])

    batch_size = array_ops.shape(inputs)[1]

    # PopnnGRU doesn't support a dynamic training parameter.
    if not isinstance(training, bool):
      raise ValueError(
          "PopnnGRU does not support a dynamic training argument.  Please pass "
          "a boolean True/False to the call method.  If you are using "
          "keras.Sequential, you should change to another model type.")

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = ops.convert_to_tensor(initial_state, dtype=dtype)

    if self._dropout > 0.:
      inputs = self._apply_dropout(inputs, training)

    output, output_state, _ = gen_popnn_ops.popnn_gru_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)

    if not self._time_major:
      # Convert output from Poplibs [S, B, N] to Keras [B, S, N]
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
      A python array.
    """
    return [batch_size, self.num_units]

  def _zero_state(self, batch_size):
    return array_ops.zeros(self.state_shape(batch_size), dtype=self.dtype)


# Alias the 2 classes without the Popnn prefix
LSTM = PopnnLSTM
GRU = PopnnGRU

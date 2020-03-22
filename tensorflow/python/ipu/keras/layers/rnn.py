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
               weights_initializer=None,
               bias_initializer=None,
               dtype=dtypes.float32,
               return_state=False,
               **kwargs):
    """Creates a _PopnnRNN model from model spec.

        Args:
          num_units: the number of units within the RNN model.
          partials_dtype: the type used by Popnn to perform partial
                          calculations.
            Either tf.float16 or tf.float32.
          seed: A Python integer. Used to create the default Glorot uniform
            initializer weights_initializer.
          weights_initializer: starting value to initialize the weight
            (default is all zeros).
          bias_initializer: starting value to initialize the bias
            (default is all zeros).
          return_state: When True, the layer returns a tuple containing the
            output and the state tensors.  Otherwise it returns only the
            output tensor.
    """
    super(_PopnnRNN, self).__init__(dtype=dtype, **kwargs)

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
    self._return_state = return_state
    # Init input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

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
          ValueError: if input_shape has wrong dimension or unknown 3rd
          dimension.
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
    if self._weights_initializer is None:
      self._weights_initializer = init_ops.glorot_uniform_initializer(
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
                                    initializer=self._weights_initializer,
                                    shape=self.canonical_weight_shape)
      self.recurrent_kernel = None
    else:
      # Initialize the input weight tensor.
      input_kernel_shape = self.canonical_weight_shape
      input_kernel_shape[0] -= self.num_units
      self.kernel = self.add_weight("kernel",
                                    dtype=self._plain_dtype,
                                    initializer=self._weights_initializer,
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
      # TODO we only support one layer.
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

    Args:
      num_units: the number of units within the RNN model.
      partials_dtype: the type used by Popnn to perform partial
                      calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      recurrent_weight_initializer=None,
      weights_initializer: starting value to initialize the weight
        (default is all zeros).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      recurrent_weight_initializer: This optional parameter will partition
                                    weight initialization into two stages,
                                    first initalizing the input kernel
                                    using weights_initializer then will
                                    initalize a kernel for the recurrent
                                    state. This partitioning is what the
                                    keras LSTM layer does.
                                    (default is None, meaning off)
      return_state: When True, the layer returns a tuple containing the
        output and the state tensors.  Otherwise it returns only the
        output tensor.
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  def __init__(self,
               units,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               recurrent_weight_initializer=None,
               return_state=False,
               **kwargs):
    if recurrent_weight_initializer is not None:
      self.recurrent_weight_initializer = initializers.get(
          recurrent_weight_initializer)
    else:
      self.recurrent_weight_initializer = None

    super(PopnnLSTM, self).__init__(num_units=units,
                                    partials_dtype=partials_dtype,
                                    seed=seed,
                                    weights_initializer=weights_initializer,
                                    bias_initializer=bias_initializer,
                                    return_state=return_state,
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
    self._build(input_shape, self.recurrent_weight_initializer)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the LSTM layer.

    Args:
      inputs: 3-D tensor with shape [time_len, batch_size, input_size].
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is initialized to
        zeros.
      training: whether this operation will be used in training or inference.

    Returns:
      tuple of output and output states:

    """
    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

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
      self.kernel = array_ops.concat([self.kernel, self.recurrent_kernel], 0)

    c, h = initial_state

    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)

    output, output_state = self._forward(inputs, h, c, self.kernel,
                                         self.biases, training)

    if self._return_state:
      return output, output_state

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

  def _forward(self, inputs, h, c, kernel, biases, training):
    output, output_h, output_c, _ = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=kernel,
        biases=biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)
    return output, rnn_cell.LSTMStateTuple(output_c, output_h)


class PopnnGRU(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an GRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      lstm = PopnnGRU(num_units, ...)

      outputs, output_state = lstm(inputs, initial_state, training=True)

  Args:
    units: the number of units within the RNN model.
    partials_dtype: the type used by Popnn to perform partial
      calculations. Either tf.float16 or tf.float32.
    seed: A Python integer. Used to create the default Glorot uniform
      initializer weights_initializer.
    weights_initializer: starting value to initialize the weight
      (default isipu Glorot uniform initializer).
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    return_state: When True, the layer returns a tuple containing the
      output and the state tensors.  Otherwise it returns only the
      output tensor.
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_GRU
  _num_gates_per_layer = POPNN_GRU_NUM_GATES

  def __init__(self,
               units,
               partials_dtype=dtypes.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               return_state=False,
               **kwargs):
    super(PopnnGRU, self).__init__(num_units=units,
                                   partials_dtype=partials_dtype,
                                   seed=seed,
                                   weights_initializer=weights_initializer,
                                   bias_initializer=bias_initializer,
                                   return_state=return_state,
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
      inputs: 3-D tensor with shape [time_len, batch_size, input_size].
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`
        If not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.

    Returns:
      output: a tensor of shape [time_len, batch_size, num_units].
      output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

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
    output, output_state = self._forward(inputs, initial_state, self.kernel,
                                         self.biases, training)

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

  def _forward(self, inputs, initial_state, kernel, biases, training):
    output, output_c, _ = gen_popnn_ops.popnn_gru_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=kernel,
        biases=biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)
    return output, output_c

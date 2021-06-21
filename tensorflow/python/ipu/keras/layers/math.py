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
Maths Keras layers
~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ipu.ops import math_ops as ipu_math_ops


class SerialDense(Layer):
  """Densely-connected NN layer where the dot operation is serialized to reduce
  the size of this operation.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Given the `input` tensor with shape `[..., m, k]` and `kernel` tensor with
  shape `[k, n]`, the matrix multiplication can be serialized as follows:

  * Along the `m` dimension of `input`, by setting `serialization_dimension` to
    `input_columns`.
  * Along the `k` dimension of `input` and `kernel` by setting
    `serialization_dimension` to `input_rows_kernel_columns`.
  * Along `n` dimension of `kernel`, by setting `serialization_dimension` to
    `kernel_rows`.

  Example:

  .. code-block:: python

    # as first layer in a sequential model:
    model = Sequential()
    model.add(SerialDense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(SerialDense(32))

  Arguments:
    units: Positive integer, dimensionality of the output space.
    serialization_factor: An integer indicating the number of smaller matrix
      multiplies this operation is broken up into. Must divide the dimension
      along which the operation is serialized on.
    serialization_dimension: A string, must be one of `input_columns`,
      `input_rows_kernel_columns` or `kernel_rows`. Indicates the dimension
      along which the operation is serialzed on.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """
  def __init__(self,
               units,
               serialization_factor,
               serialization_dimension,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.serialization_factor = int(serialization_factor)
    self.serialization_dimension = serialization_dimension

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `SerialDense` layer with non-floating '
                      'point dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `SerialDense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight('kernel',
                                  shape=[last_dim, self.units],
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  dtype=self.dtype,
                                  trainable=True)
    if self.use_bias:
      self.bias = self.add_weight('bias',
                                  shape=[
                                      self.units,
                                  ],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  dtype=self.dtype,
                                  trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, **kwargs):
    """
    Args:
      inputs: The tensor to apply the dense weights to.

    Returns:
      The tensor resulting from applying the dense weights.
    """
    if K.is_sparse(inputs):
      raise TypeError(
          'Unable to build `SerialDense` layer with sparse inputs.')

    if self.serialization_factor < 1:
      raise ValueError(
          'serialization_factor has to be at least 1, but was {}.'.format(
              self.serialization_factor))

    inputs = math_ops.cast(inputs, self._compute_dtype)

    # Transform the dimension name.
    serialization_dimension = self.serialization_dimension
    if serialization_dimension == "input_columns":
      serialization_dimension = "a_columns"
    elif serialization_dimension == "input_rows_kernel_columns":
      serialization_dimension = "a_rows_b_columns"
    elif serialization_dimension == "kernel_rows":
      serialization_dimension = "b_rows"
    else:
      raise ValueError('Invalid serialization_dimension={}, expected one of: '
                       '\'input_columns\', \'input_rows_kernel_columns\', '
                       '\'kernel_rows\'.'.format(serialization_dimension))

    outputs = ipu_math_ops.serialized_matmul(inputs, self.kernel,
                                             self.serialization_factor,
                                             serialization_dimension)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'serialization_factor': self.serialization_factor,
        'serialization_dimension': self.serialization_dimension,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
        regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

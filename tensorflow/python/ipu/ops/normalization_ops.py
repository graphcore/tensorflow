# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""
Popnn normalization operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

# This implementation is based on:
# tensorflow/contrib/layers/python/layers/normalization.py
__all__ = [
    'group_norm',
    'instance_norm',
    'layer_norm',
]

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


def _get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections.get(name, None)
  else:
    variable_collections = variables_collections
  return variable_collections


def _group_norm_impl(inputs,
                     groups=2,
                     channels_axis=-1,
                     center=True,
                     scale=True,
                     epsilon=1.53e-5,
                     param_initializers=None,
                     reuse=None,
                     variables_collections=None,
                     training=True,
                     trainable=True,
                     scope=None,
                     norm_type="",
                     strided_channel_grouping=True):
  """Internal implementation of any group norm type operation."""

  inputs = ops.convert_to_tensor(inputs)

  if inputs.shape.ndims is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if channels_axis > (inputs.shape.ndims - 1):
    raise ValueError('Axis is out of bounds.')

  # Standardize the channels_axis to be positive and identify # of channels.
  if channels_axis < 0:
    channels_axis = inputs.shape.ndims + channels_axis
  channels = inputs.shape.as_list()[channels_axis]

  if channels_axis == 1:
    data_format = DATA_FORMAT_NCHW
  elif channels_axis == inputs.shape.ndims - 1:
    data_format = DATA_FORMAT_NHWC
  else:
    raise ValueError('Unsupported data format, group norm only supports NCHW'
                     '(channel axis 1) and NHWC (channel axis -1).')

  if channels is None:
    raise ValueError('Inputs %s has undefined channel dimension: %d.' %
                     (inputs.name, channels_axis))

  if groups > channels:
    raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
  if channels % groups != 0:
    raise ValueError('%d channels is not commensurate with %d groups.' %
                     (channels, groups))

  with variable_scope.variable_scope(scope, norm_type, [inputs], reuse=reuse):
    # Note that the params_shape is the number of channels always.
    params_shape = [channels]

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}
    if center:
      beta_collections = _get_variable_collections(variables_collections,
                                                   'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variable_scope.get_variable('beta',
                                         shape=params_shape,
                                         dtype=dtype,
                                         initializer=beta_initializer,
                                         collections=beta_collections,
                                         trainable=trainable)
    else:
      beta = array_ops.constant(0.0, dtype=dtype, shape=params_shape)

    if scale:
      gamma_collections = _get_variable_collections(variables_collections,
                                                    'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variable_scope.get_variable('gamma',
                                          shape=params_shape,
                                          dtype=dtype,
                                          initializer=gamma_initializer,
                                          collections=gamma_collections,
                                          trainable=trainable)
    else:
      gamma = array_ops.constant(1.0, dtype=dtype, shape=params_shape)

    if training:
      outputs, _, _ = gen_popnn_ops.popnn_group_norm_training(
          inputs=inputs,
          gamma=gamma,
          beta=beta,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups,
          strided_channel_grouping=strided_channel_grouping)

    else:
      # Calculate the moments.
      mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
          inputs=inputs,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups,
          strided_channel_grouping=strided_channel_grouping)

      outputs = gen_popnn_ops.popnn_group_norm_inference(
          inputs=inputs,
          gamma=gamma,
          beta=beta,
          mean=mean,
          inv_std_dev=inv_std_dev,
          data_format=data_format,
          epsilon=epsilon,
          num_groups=groups,
          strided_channel_grouping=strided_channel_grouping)

    return outputs


def group_norm(inputs,
               groups=2,
               channels_axis=-1,
               center=True,
               scale=True,
               epsilon=1.53e-5,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               training=True,
               trainable=True,
               scope=None,
               strided_channel_grouping=True):
  """Functional interface for the group normalization layer.

  Reference: https://arxiv.org/abs/1803.08494.

    "Group Normalization", Yuxin Wu, Kaiming He

  Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
     shape dimensions must be fully defined.
    groups: Integer. Divide the channels into this number of groups over which
      normalization statistics are computed. This number must be commensurate
      with the number of channels in `inputs`.
    channels_axis: An integer. Specifies index of channels axis which will be
      broken into `groups`, each of which whose statistics will be computed
      across. Preferred usage is to specify negative integers to be agnostic as
      to whether a batch dimension is included.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    param_initializers: Optional initializers for beta and gamma.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    training: Whether this is operation is being used in a training network.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.
    strided_channel_grouping: Selects whether to group the channels dimension
      for group normalisation with a stride between channels. Enabling this
      makes the PopLibs implementation more efficient but is unconventional.
      Among other things this will mean that using pre-trained weights would not
      be possible if not produced with this unconventional implementation.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
    ValueError: If channels dimension is not 1 or 3.
    ValueError: If number of groups is not commensurate with number of channels.
  """
  if epsilon < 1.53e-5:
    logging.warning(
        'The epsilon value of group_norm is too low, which can lead to '
        'NaN values or floating point exceptions. To avoid this, increase '
        'it to a value higher than 1.53e-5.')

  return _group_norm_impl(inputs, groups, channels_axis, center, scale,
                          epsilon, param_initializers, reuse,
                          variables_collections, training, trainable, scope,
                          "GroupNorm", strided_channel_grouping)


def layer_norm(inputs,
               channels_axis=-1,
               center=True,
               scale=True,
               epsilon=1.53e-5,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               training=True,
               trainable=True,
               scope=None):
  """Adds a Layer Normalization layer.

  Based on the paper:

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    https://arxiv.org/abs/1607.06450.

  Layer normalization will generate normalization statistics across the
  spatial (X,Y,...) dimensions and the feature channels dimension (C). It is
  equivalent to a group normalization where all of the features in the feature
  channels dimension are put into a single group.

  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
  and this part of the inputs' shape must be fully defined.

  Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
     shape dimensions must be fully defined.
    channels_axis: An integer. Specifies index of channels axis. Preferred
      usage is to specify negative integers to be agnostic as to whether a
      batch dimension is included.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    param_initializers: Optional initializers for beta and gamma.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    training: Whether this is operation is being used in a training network.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation, having the same
    shape and dtype as `inputs`.

  Raises:
    ValueError: If the rank of `inputs` is not known at graph build time,
      or if `inputs.shape[begin_params_axis:]` is not fully defined at
      graph build time.
  """
  if inputs.shape.ndims is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if channels_axis > (inputs.shape.ndims - 1):
    raise ValueError('Axis is out of bounds.')

  if epsilon < 1.53e-5:
    logging.warning(
        'The epsilon value of layer_norm is too low, which can lead to '
        'NaN values or floating point exceptions. To avoid this, increase '
        'it to a value higher than 1.53e-5.')

  groups = 1

  return _group_norm_impl(inputs, groups, channels_axis, center, scale,
                          epsilon, param_initializers, reuse,
                          variables_collections, training, trainable, scope,
                          "LayerNorm", False)


def instance_norm(inputs,
                  channels_axis=-1,
                  center=True,
                  scale=True,
                  epsilon=1.53e-5,
                  param_initializers=None,
                  reuse=None,
                  variables_collections=None,
                  training=True,
                  trainable=True,
                  scope=None):
  """Functional interface for the instance normalization layer.

  Reference: https://arxiv.org/abs/1607.08022.

    "Instance Normalization: The Missing Ingredient for Fast Stylization"
    Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

  Instance normalization will generate normalization statistics across the
  spatial (X,Y,...) dimensions.  Each slice along the feature channels
  dimension (C) is normalized independently. It is equivalent to a group
  normalization where the number of groups is the same as the size of the
  feature channels dimension.

  Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
      shape dimensions must be fully defined.
    channels_axis: An integer. Specifies index of channels axis. Preferred
      usage is to specify negative integers to be agnostic as to whether a
      batch dimension is included.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    param_initializers: Optional initializers for beta and gamma.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    training: Whether this is operation is being used in a training network.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if inputs.shape.ndims is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if channels_axis > (inputs.shape.ndims - 1):
    raise ValueError('Axis is out of bounds.')

  if epsilon < 1.53e-5:
    logging.warning(
        'The epsilon value of instance_norm is too low, which can lead to '
        'NaN values or floating point exceptions. To avoid this, increase '
        'it to a value higher than 1.53e-5.')

  if channels_axis < 0:
    channels_axis = inputs.shape.ndims + channels_axis
  groups = inputs.shape.as_list()[channels_axis]

  return _group_norm_impl(inputs, groups, channels_axis, center, scale,
                          epsilon, param_initializers, reuse,
                          variables_collections, training, trainable, scope,
                          "InstanceNorm", False)

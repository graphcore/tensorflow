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
"""Tests for IPU Norm layers."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python import ipu
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python import keras

dataType = np.float32


def keras_instance(instance, x_val, training=True, **kwargs):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    output = ipu.layers.InstanceNorm(**kwargs)(inputs=x, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val})


def keras_layer(instance, x_val, training=True, **kwargs):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    output = ipu.layers.LayerNorm(**kwargs)(inputs=x, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val})


def keras_upstream_layer(x, training=True, **kwargs):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    output = keras.layers.LayerNormalization(**kwargs)(inputs=x,
                                                       training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val})


def keras_layer_copy_weights(instance, input_shape, **kwargs):
  with ops.device('/device:IPU:0'):
    layer = ipu.layers.LayerNormalization(**kwargs)
    upstream_layer = keras.layers.LayerNormalization(**kwargs)
    layer.build(input_shape)
    upstream_layer.build(input_shape)
    layer.set_weights(upstream_layer.get_weights())

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(
        (layer.beta, layer.gamma, upstream_layer.beta, upstream_layer.gamma))


def keras_group(instance, x_val, training=True, **kwargs):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    output = ipu.layers.GroupNorm(**kwargs)(inputs=x, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val})


class GroupNorm(test.TestCase):
  def doOutputTest(self,
                   input_shape,
                   channels_axis=None,
                   reduction_axes=None,
                   groups=2,
                   tol=1e-1):
    # Select the axis for the channel and the dimensions along which statistics
    # are accumulated.
    if channels_axis < 0:
      channels_axis += len(input_shape)
    reduced_axes = [channels_axis + 1]
    for a in reduction_axes:
      if a < 0:
        a += len(input_shape)
      if a < channels_axis:
        reduced_axes.append(a)
      else:
        reduced_axes.append(a + 1)
    reduced_axes = tuple(reduced_axes)
    channels = input_shape[channels_axis]
    group_size = channels // groups
    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis + 1:]
    outputs_shape = (axes_before_channels + [1, channels] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    mu = 1.0
    sigma = 1.0
    # Determine shape of Tensor after normalization.
    expected_mean = np.zeros(reduced_shape)
    expected_var = np.ones(reduced_shape)

    inputs = np.random.rand(*input_shape).astype(dataType) * sigma + mu
    outputs = keras_group(self,
                          inputs,
                          groups=groups,
                          center=False,
                          scale=False,
                          channels_axis=channels_axis,
                          training=True)

    # Make sure that there are no NaNs
    self.assertFalse(np.isnan(outputs).any())

    # Implementation detail - in Poplibs group norm, the groups are not
    # contiguous, but strided - we replicate that here
    # Move the channels to the first dimension for inputs, gamma and beta
    outputs = np.swapaxes(outputs, 0, channels_axis)
    reshuffled_outputs = np.empty(outputs.shape, outputs.dtype)
    for from_idx in range(channels):
      to_idx = (from_idx % groups) * group_size + from_idx // groups
      reshuffled_outputs[to_idx] = outputs[from_idx]
    outputs = np.swapaxes(reshuffled_outputs, 0, channels_axis)

    outputs = np.reshape(outputs, outputs_shape)
    mean = np.mean(outputs, axis=reduced_axes, dtype=np.float32)
    var = np.var(outputs, axis=reduced_axes, dtype=np.float32)
    # The mean and variance of each example should be close to 0 and 1
    # respectively.
    self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
    self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput2D_NC(self):
    self.doOutputTest([10, 7 * 100],
                      channels_axis=1,
                      reduction_axes=[],
                      groups=7)

  @test_util.deprecated_graph_mode_only
  def testOutput5D_NCXXX(self):
    self.doOutputTest([4, 4, 4, 10, 4],
                      channels_axis=1,
                      reduction_axes=[2, 3, 4],
                      groups=2)


class LayerTest(test.TestCase):
  def doTest(self,
             input_shape,
             channels_axis=None,
             reduction_axes=None,
             tol=1e-1):
    # Select the axis for the channel and the dimensions along which statistics
    # are accumulated.
    if channels_axis < 0:
      channels_axis += len(input_shape)
    reduced_axes = [channels_axis + 1]
    axis = [channels_axis]
    for a in reduction_axes:
      if a < 0:
        a += len(input_shape)
      if a < channels_axis:
        reduced_axes.append(a)
      else:
        reduced_axes.append(a + 1)
      axis.append(a)
    reduced_axes = tuple(reduced_axes)
    channels = input_shape[channels_axis]
    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis + 1:]
    outputs_shape = (axes_before_channels + [1, channels] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    mu = 1.0
    sigma = 1.0
    # Determine shape of Tensor after normalization.
    expected_mean = np.zeros(reduced_shape)
    expected_var = np.ones(reduced_shape)

    inputs = np.random.rand(*input_shape).astype(dataType) * sigma + mu
    result = keras_layer(self,
                         inputs,
                         center=False,
                         scale=False,
                         axis=axis,
                         training=True)
    # Make sure that there are no NaNs
    self.assertFalse(np.isnan(result).any())

    result = np.swapaxes(result, 0, channels_axis)
    result = np.reshape(result, outputs_shape)
    mean = np.mean(result, axis=reduced_axes, dtype=np.float32)
    var = np.var(result, axis=reduced_axes, dtype=np.float32)
    # The mean and variance of each example should be close to 0 and 1
    # respectively.
    self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
    self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput2D_NC(self):
    self.doTest([10, 7 * 100], channels_axis=1, reduction_axes=[])

  @test_util.deprecated_graph_mode_only
  def testOutput5D_NCXXX(self):
    self.doTest([4, 4, 4, 10, 4], channels_axis=1, reduction_axes=[2, 3, 4])

  def doComparisonTest(self, input_shape, axis):
    inputs = np.random.rand(*input_shape).astype(dataType) + 0.1
    result = keras_layer(self, inputs, axis=axis)
    result_upstream = keras_upstream_layer(inputs, axis=axis)
    self.assertAllClose(result, result_upstream, rtol=1e-3)

  @test_util.run_v2_only
  def test3D(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doComparisonTest(input_shape, axis=[1, 2])
    # Specify axes with negative values.
    self.doComparisonTest(input_shape, axis=[-2, -1])

  @test_util.run_v2_only
  def test4D_single_axis(self):
    input_shape = [10, 10, 30, 10]
    # Specify axes with positive values.
    self.doComparisonTest(input_shape, axis=[2])
    # Specify axes with negative values.
    self.doComparisonTest(input_shape, axis=[-1])

  def testCopyWeightsFromUpstreamLayer(self):
    input_shape = (10, 10, 30)
    axis = (-1)
    # Build upstream layer and copy its weights to the ipu layer.
    layer_beta, layer_gamma, upstream_layer_beta, upstream_layer_gamma = \
      keras_layer_copy_weights(self, input_shape, axis=axis)
    self.assertAllEqual(layer_beta, upstream_layer_beta)
    self.assertAllEqual(layer_gamma, upstream_layer_gamma)


class InstanceTest(test.TestCase):
  def doTest(self,
             input_shape,
             channels_axis=None,
             reduction_axes=None,
             tol=1e-1):
    # Select the axis for the channel and the dimensions along which statistics
    # are accumulated.
    if channels_axis < 0:
      channels_axis += len(input_shape)
    reduced_axes = [channels_axis + 1]
    for a in reduction_axes:
      if a < 0:
        a += len(input_shape)
      if a < channels_axis:
        reduced_axes.append(a)
      else:
        reduced_axes.append(a + 1)
    reduced_axes = tuple(reduced_axes)
    channels = input_shape[channels_axis]
    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis + 1:]
    outputs_shape = (axes_before_channels + [channels, 1] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    mu = 1.0
    sigma = 1.0
    # Determine shape of Tensor after normalization.
    expected_mean = np.zeros(reduced_shape)
    expected_var = np.ones(reduced_shape)

    inputs = np.random.rand(*input_shape).astype(dataType) * sigma + mu
    outputs = keras_instance(self,
                             inputs,
                             center=False,
                             scale=False,
                             channels_axis=channels_axis,
                             training=True)

    # Implementation detail - in Poplibs group norm, the groups are not
    # contiguous, but strided - we replicate that here
    # Move the channels to the first dimension for inputs, gamma and beta
    outputs = np.swapaxes(outputs, 0, channels_axis)
    reshuffled_outputs = np.empty(outputs.shape, outputs.dtype)
    for from_idx in range(channels):
      to_idx = (from_idx % channels) + from_idx // channels
      reshuffled_outputs[to_idx] = outputs[from_idx]
    outputs = np.swapaxes(reshuffled_outputs, 0, channels_axis)

    outputs = np.reshape(outputs, outputs_shape)
    mean = np.mean(outputs, axis=reduced_axes, dtype=np.float32)
    var = np.var(outputs, axis=reduced_axes, dtype=np.float32)
    # The mean and variance of each example should be close to 0 and 1
    # respectively.
    self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
    self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  @test_util.deprecated_graph_mode_only
  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  @test_util.deprecated_graph_mode_only
  def testOutput5D_NCXXX(self):
    self.doTest([4, 4, 4, 10, 4], channels_axis=1, reduction_axes=[2, 3, 4])


if __name__ == '__main__':
  test.main()

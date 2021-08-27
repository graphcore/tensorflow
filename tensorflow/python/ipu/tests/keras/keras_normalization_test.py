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

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.framework import test_util

dataType = np.float32


def keras_instance(x, training=True, **kwargs):
  layer = ipu.layers.InstanceNormalization(**kwargs)
  layer.build(x.shape)

  @def_function.function
  def impl(x, training):
    return layer(inputs=x, training=training)

  return impl(x, training)


def keras_layer(x, training=True, **kwargs):
  layer = ipu.layers.LayerNormalization(**kwargs)
  layer.build(x.shape)

  @def_function.function
  def impl(x, training):
    return layer(inputs=x, training=training)

  return impl(x, training)


def keras_upstream_layer(x, training=True, **kwargs):
  layer = keras.layers.LayerNormalization(**kwargs)
  layer.build(x.shape)

  @def_function.function
  def impl(x, training):
    return layer(inputs=x, training=training)

  return impl(x, training)


def keras_group(x, training=True, **kwargs):
  layer = ipu.layers.GroupNormalization(**kwargs)
  layer.build(x.shape)

  @def_function.function
  def impl(x, training):
    return layer(inputs=x, training=training)

  return impl(x, training)


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
    outputs = keras_group(inputs,
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

  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  def testOutput2D_NC(self):
    self.doOutputTest([10, 7 * 100],
                      channels_axis=1,
                      reduction_axes=[],
                      groups=7)

  def testOutput5D_NCXXX(self):
    self.doOutputTest([4, 4, 4, 10, 4],
                      channels_axis=1,
                      reduction_axes=[2, 3, 4],
                      groups=2)

  def testDtype(self):
    layer = ipu.layers.GroupNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float32 for w in layer.weights))

    layer = ipu.layers.GroupNormalization(dtype=dtypes.float16)
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

    keras.backend.set_floatx('float16')
    layer = ipu.layers.GroupNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

  @test_util.run_v2_only
  def testGetConfig(self):
    layer = ipu.layers.GroupNormalization()
    config = layer.get_config()
    layer2 = ipu.layers.GroupNormalization.from_config(config)
    self.assertEqual(config, layer2.get_config())


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
    result = keras_layer(inputs,
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

  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  def testOutput2D_NC(self):
    self.doTest([10, 7 * 100], channels_axis=1, reduction_axes=[])

  def testOutput5D_NCXXX(self):
    self.doTest([4, 4, 4, 10, 4], channels_axis=1, reduction_axes=[2, 3, 4])

  def doComparisonTest(self, input_shape, axis):
    inputs = np.random.rand(*input_shape).astype(dataType) + 0.1
    result = keras_layer(inputs, axis=axis)
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

  def testDtype(self):
    layer = ipu.layers.LayerNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float32 for w in layer.weights))

    layer = ipu.layers.LayerNormalization(dtype=dtypes.float16)
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

    keras.backend.set_floatx('float16')
    layer = ipu.layers.LayerNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

  @test_util.run_v2_only
  def testGetConfig(self):
    layer = ipu.layers.LayerNormalization()
    config = layer.get_config()
    layer2 = ipu.layers.LayerNormalization.from_config(config)
    self.assertEqual(config, layer2.get_config())


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
    outputs = keras_instance(inputs,
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

  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  def testOutput5D_NCXXX(self):
    self.doTest([4, 4, 4, 10, 4], channels_axis=1, reduction_axes=[2, 3, 4])

  def testDtype(self):
    layer = ipu.layers.InstanceNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float32 for w in layer.weights))

    layer = ipu.layers.InstanceNormalization(dtype=dtypes.float16)
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

    keras.backend.set_floatx('float16')
    layer = ipu.layers.InstanceNormalization()
    layer.build((1, 1, 1, 2))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer.weights))

  @test_util.run_v2_only
  def testGetConfig(self):
    layer = ipu.layers.InstanceNormalization()
    config = layer.get_config()
    layer2 = ipu.layers.InstanceNormalization.from_config(config)
    self.assertEqual(config, layer2.get_config())


if __name__ == '__main__':
  test.main()

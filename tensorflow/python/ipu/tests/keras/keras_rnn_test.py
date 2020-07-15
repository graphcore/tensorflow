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
"""Tests for IPU LSTM layers."""

import numpy as np

from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables

from tensorflow.python import ipu

# Test hyperparameters.
batch_size = 1
num_input = 3
timesteps = 4
num_hidden = 5
data_type = np.float32


def _kerasLSTMImpl(instance,
                   x_vals,
                   h_val,
                   c_val,
                   keras_layer=None,
                   device="cpu",
                   training=True,
                   return_state=True,
                   return_sequences=False,
                   time_major=False,
                   dropout=0.,
                   unit_forget_bias=False,
                   stateful=False):

  with ops.device(device):
    x = array_ops.placeholder(x_vals[0].dtype, x_vals[0].shape)
    h = array_ops.placeholder(h_val.dtype, h_val.shape)
    c = array_ops.placeholder(c_val.dtype, c_val.shape)

    state = None if stateful else rnn_cell.LSTMStateTuple(c, h)

    output = keras_layer(
        num_hidden,
        dtype=data_type,
        kernel_initializer=init_ops.constant_initializer(0.1, data_type),
        recurrent_initializer=init_ops.constant_initializer(0.2, data_type),
        bias_initializer=init_ops.constant_initializer(0.3, data_type),
        recurrent_activation="sigmoid",
        dropout=dropout,
        time_major=time_major,
        return_sequences=return_sequences,
        return_state=return_state,
        unit_forget_bias=unit_forget_bias,
        stateful=stateful)(inputs=x, initial_state=state, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []
    for x_val in x_vals:
      outputs.append(sess.run(output, {x: x_val, h: h_val, c: c_val}))
    return outputs


def _lstmIPU(*args, **kwargs):
  return _kerasLSTMImpl(*args,
                        **kwargs,
                        keras_layer=ipu.layers.PopnnLSTM,
                        device='/device:IPU:0')


def _lstmCPU(*args, **kwargs):
  return _kerasLSTMImpl(*args, **kwargs, keras_layer=recurrent_v2.LSTM)


class IpuLstmTest(test.TestCase):
  def _get_random_inputs(self, time_major=False, num_samples=1):
    np.random.seed(42)
    h = np.random.rand(batch_size, num_hidden).astype(data_type)
    c = np.random.rand(batch_size, num_hidden).astype(data_type)
    xs = []
    for _ in range(num_samples):
      shape = [timesteps, batch_size, num_input] \
              if time_major else [batch_size, timesteps, num_input]
      xs.append(np.random.rand(*shape).astype(data_type))
    return xs, h, c

  @test_util.deprecated_graph_mode_only
  def test_lstm(self):
    x, h, c = self._get_random_inputs()

    cpu_result = _lstmCPU(self, x, h, c)
    ipu_result = _lstmIPU(self, x, h, c)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_time_major(self):
    x, h, c = self._get_random_inputs(time_major=True)

    cpu_result = _lstmCPU(self, x, h, c, time_major=True)
    ipu_result = _lstmIPU(self, x, h, c, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_unit_forget_bias(self):
    x, h, c = self._get_random_inputs()

    cpu_result = _lstmCPU(self, x, h, c, unit_forget_bias=True)
    ipu_result = _lstmIPU(self, x, h, c, unit_forget_bias=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_all_seq(self):
    x, h, c = self._get_random_inputs()

    ipu_result = _lstmIPU(self, x, h, c, return_sequences=True)
    cpu_result = _lstmCPU(self, x, h, c, return_sequences=True)
    self.assertAllClose(ipu_result, cpu_result)

    self.assertEqual(ipu_result[0][0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_no_state(self):
    x, h, c = self._get_random_inputs()

    ipu_result = _lstmIPU(self, x, h, c, return_state=False)
    self.assertTrue(isinstance(ipu_result[0], np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_no_dynamic_training(self):
    x, h, c = self._get_random_inputs()

    with self.assertRaisesRegex(ValueError,
                                'PopnnLSTM does not support a dynamic'):
      _lstmIPU(self, x, h, c, training=None)

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(ipu.layers.LSTM, type))
    self.assertEqual(ipu.layers.PopnnLSTM, ipu.layers.LSTM)

  @test_util.deprecated_graph_mode_only
  def test_lstm_dropout(self):
    x, h, c = self._get_random_inputs()

    dropout_none_result = _lstmIPU(self,
                                   x,
                                   h,
                                   c,
                                   return_state=False,
                                   dropout=0.)
    dropout_most_result = _lstmIPU(self,
                                   x,
                                   h,
                                   c,
                                   return_state=False,
                                   dropout=0.9)

    self.assertNotAllClose(dropout_none_result, dropout_most_result)

  @test_util.run_v2_only
  def test_can_call_without_state_change(self):
    x, h, c = self._get_random_inputs()

    layer = ipu.layers.PopnnLSTM(
        num_hidden,
        dtype=data_type,
        kernel_initializer=init_ops.random_uniform_initializer(
            seed=42, dtype=data_type),
        recurrent_initializer=init_ops.random_uniform_initializer(
            seed=42, dtype=data_type),
        bias_initializer=init_ops.zeros_initializer(dtype=data_type))
    layer.build(x.shape)

    @def_function.function
    def impl(x, c, h):
      state = rnn_cell.LSTMStateTuple(c, h)
      return layer(inputs=x, initial_state=state)

    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x, c, h)
    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x, c, h)


def _kerasGRUImpl(instance,
                  x_vals,
                  init_val,
                  keras_layer=None,
                  device="cpu",
                  training=True,
                  return_state=True,
                  return_sequences=False,
                  time_major=False,
                  dropout=0.,
                  stateful=False):

  with ops.device(device):
    x = array_ops.placeholder(x_vals[0].dtype, x_vals[0].shape)
    init = array_ops.placeholder(init_val.dtype, init_val.shape)

    output = keras_layer(
        num_hidden,
        dtype=data_type,
        kernel_initializer=init_ops.constant_initializer(0.1, data_type),
        recurrent_initializer=init_ops.constant_initializer(0.2, data_type),
        bias_initializer=init_ops.constant_initializer(0.3, data_type),
        recurrent_activation="sigmoid",
        dropout=dropout,
        time_major=time_major,
        return_sequences=return_sequences,
        return_state=return_state,
        reset_after=False,
        stateful=stateful)(inputs=x, initial_state=init, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []
    for x_val in x_vals:
      outputs.append(sess.run(output, {x: x_val, init: init_val}))
    return outputs


def _gruIPU(*args, **kwargs):
  return _kerasGRUImpl(*args,
                       **kwargs,
                       keras_layer=ipu.layers.PopnnGRU,
                       device='/device:IPU:0')


def _gruCPU(*args, **kwargs):
  return _kerasGRUImpl(*args, **kwargs, keras_layer=recurrent_v2.GRU)


class IpuGruTest(test.TestCase):
  def _get_random_inputs(self, time_major=False, num_samples=1):
    np.random.seed(43)
    init = np.random.rand(batch_size, num_hidden).astype(data_type)
    xs = []
    for _ in range(num_samples):
      shape = [timesteps, batch_size, num_input] \
              if time_major else [batch_size, timesteps, num_input]
      xs.append(np.random.rand(*shape).astype(data_type))
    return xs, init

  @test_util.deprecated_graph_mode_only
  def test_gru(self):
    x, init = self._get_random_inputs()

    cpu_result = _gruCPU(self, x, init)
    ipu_result = _gruIPU(self, x, init)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_seq_major(self):
    x, init = self._get_random_inputs(True)

    ipu_result = _gruIPU(self, x, init, time_major=True)
    cpu_result = _gruCPU(self, x, init, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_all_seq(self):
    x, init = self._get_random_inputs()

    ipu_result = _gruIPU(self, x, init, return_sequences=True)
    cpu_result = _gruCPU(self, x, init, return_sequences=True)

    self.assertAllClose(ipu_result, cpu_result)
    self.assertEqual(ipu_result[0][0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_gru_no_state(self):
    x, init = self._get_random_inputs()

    ipu_result = _gruIPU(self, x, init, return_state=False)
    self.assertTrue(isinstance(ipu_result[0], np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_no_dynamic_training(self):
    x, init = self._get_random_inputs()

    with self.assertRaisesRegex(ValueError,
                                'PopnnGRU does not support a dynamic'):
      _gruIPU(self, x, init, training=None)

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(ipu.layers.GRU, type))
    self.assertEqual(ipu.layers.PopnnGRU, ipu.layers.GRU)

  @test_util.deprecated_graph_mode_only
  def test_gru_dropout(self):
    x, init = self._get_random_inputs()

    dropout_none_result = _gruIPU(self,
                                  x,
                                  init,
                                  dropout=0.,
                                  return_state=False,
                                  return_sequences=True)
    dropout_most_result = _gruIPU(self,
                                  x,
                                  init,
                                  dropout=0.9,
                                  return_state=False,
                                  return_sequences=True)

    self.assertNotAllClose(dropout_none_result, dropout_most_result)


if __name__ == '__main__':
  test.main()

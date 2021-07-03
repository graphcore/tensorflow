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

from tensorflow.python import keras
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


def _getLSTMLayer(keras_layer=None,
                  return_state=True,
                  return_sequences=False,
                  time_major=False,
                  dropout=0.,
                  unit_forget_bias=False,
                  stateful=False,
                  kernel_initializer=None,
                  recurrent_initializer=None,
                  bias_initializer=None):
  kernel_initializer = (kernel_initializer if kernel_initializer else
                        init_ops.constant_initializer(0.1, data_type))
  recurrent_initializer = (recurrent_initializer if recurrent_initializer else
                           init_ops.constant_initializer(0.2, data_type))
  bias_initializer = (bias_initializer if bias_initializer else
                      init_ops.constant_initializer(0.3, data_type))
  return keras_layer(num_hidden,
                     dtype=data_type,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     recurrent_activation="sigmoid",
                     dropout=dropout,
                     time_major=time_major,
                     return_sequences=return_sequences,
                     return_state=return_state,
                     unit_forget_bias=unit_forget_bias,
                     stateful=stateful)


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

    layer = _getLSTMLayer(keras_layer, return_state, return_sequences,
                          time_major, dropout, unit_forget_bias, stateful)
    output = layer(inputs=x, initial_state=state, training=training)
    shapes = [w.shape for w in layer.get_weights()]

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []

    # Run the op and any updates.
    to_run = [output, [layer.updates]] if layer.updates else output
    for x_val in x_vals:
      r = sess.run(to_run, {x: x_val, h: h_val, c: c_val})
      r = r[0] if layer.updates else r
      outputs.append(r)
    return (outputs, shapes)


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

    self.assertEqual(ipu_result[0][0][0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_no_state(self):
    x, h, c = self._get_random_inputs()

    ipu_result = _lstmIPU(self, x, h, c, return_state=False)
    self.assertTrue(isinstance(ipu_result[0][0], np.ndarray))

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
    layer.build(x[0].shape)

    @def_function.function
    def impl(x, c, h):
      state = rnn_cell.LSTMStateTuple(c, h)
      return layer(inputs=x, initial_state=state)

    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x[0], c, h)
    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x[0], c, h)

  @test_util.deprecated_graph_mode_only
  def test_lstm_stateful(self):
    x, h, c = self._get_random_inputs(num_samples=10)

    cpu_result = _lstmCPU(self, x, h, c, stateful=True)
    ipu_result = _lstmIPU(self, x, h, c, stateful=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_stateful_time_major(self):
    x, h, c = self._get_random_inputs(time_major=True, num_samples=10)

    cpu_result = _lstmCPU(self, x, h, c, stateful=True, time_major=True)
    ipu_result = _lstmIPU(self, x, h, c, stateful=True, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_lstm_save_load_weights(self):
    xs, _, _ = self._get_random_inputs()
    x = xs[0]
    # Run on CPU
    layer_cpu = _getLSTMLayer(recurrent_v2.LSTM,
                              kernel_initializer='truncated_normal',
                              recurrent_initializer='normal',
                              bias_initializer='truncated_normal')
    cpu_result = layer_cpu(x, training=True)

    # Create IPU layer, build it, and get the weights from the cpu layer.
    layer_ipu = _getLSTMLayer(ipu.layers.PopnnLSTM)
    layer_ipu.build((batch_size, timesteps, num_input))
    layer_ipu.set_weights(layer_cpu.get_weights())

    ipu_result = layer_ipu(x, training=True)
    self.assertAllClose(ipu_result, cpu_result)


def _getGRULayer(keras_layer=None,
                 return_state=True,
                 return_sequences=False,
                 time_major=False,
                 dropout=0.,
                 stateful=False,
                 reset_after=False,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer=None):
  kernel_initializer = (kernel_initializer
                        or init_ops.constant_initializer(0.1, data_type))
  recurrent_initializer = (recurrent_initializer
                           or init_ops.constant_initializer(0.2, data_type))
  bias_initializer = (bias_initializer
                      or init_ops.constant_initializer(0.3, data_type))
  return keras_layer(num_hidden,
                     dtype=data_type,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     recurrent_activation="sigmoid",
                     dropout=dropout,
                     time_major=time_major,
                     return_sequences=return_sequences,
                     return_state=return_state,
                     reset_after=reset_after,
                     stateful=stateful)


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
                  stateful=False,
                  reset_after=False):

  with ops.device(device):
    x = array_ops.placeholder(x_vals[0].dtype, x_vals[0].shape)
    init_ph = array_ops.placeholder(init_val.dtype, init_val.shape)

    init = None if stateful else init_ph

    layer = _getGRULayer(keras_layer, return_state, return_sequences,
                         time_major, dropout, stateful, reset_after)
    output = layer(inputs=x, initial_state=init, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []

    # Run the op and any updates.
    to_run = [output, [layer.updates]] if layer.updates else output
    for x_val in x_vals:
      r = sess.run(to_run, {x: x_val, init_ph: init_val})
      r = r[0] if layer.updates else r
      outputs.append(r)
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

  @test_util.deprecated_graph_mode_only
  def test_gru_stateful(self):
    x, init = self._get_random_inputs(num_samples=10)

    cpu_result = _gruCPU(self, x, init, stateful=True)
    ipu_result = _gruIPU(self, x, init, stateful=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_stateful_time_major(self):
    x, init = self._get_random_inputs(time_major=True, num_samples=10)

    cpu_result = _gruCPU(self, x, init, stateful=True, time_major=True)
    ipu_result = _gruIPU(self, x, init, stateful=True, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_reset_after(self):
    x, init = self._get_random_inputs(num_samples=10)

    cpu_result = _gruCPU(self, x, init, reset_after=True)
    ipu_result = _gruIPU(self, x, init, reset_after=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_gru_save_load_weights(self):
    xs, _ = self._get_random_inputs()
    x = xs[0]

    # Run on CPU
    layer_cpu = _getGRULayer(recurrent_v2.GRU,
                             kernel_initializer='truncated_normal',
                             recurrent_initializer='normal',
                             bias_initializer='truncated_normal')
    cpu_result = layer_cpu(x, training=True)

    # Create IPU layer, build it, and get the weights from the cpu layer.
    layer_ipu = _getGRULayer(ipu.layers.PopnnGRU)
    layer_ipu.build((batch_size, timesteps, num_input))
    layer_ipu.set_weights(layer_cpu.get_weights())

    ipu_result = layer_ipu(x, training=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_gru_ipu_vs_cpu_results_reset_after(self):
    # Prepare Data
    xs, init = self._get_random_inputs(num_samples=2)
    x_fit, x_predict = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    # Setup CPU GRU layer
    layer_cpu = _getGRULayer(recurrent_v2.GRU,
                             kernel_initializer='truncated_normal',
                             recurrent_initializer='normal',
                             bias_initializer='truncated_normal',
                             reset_after=True,
                             return_state=False)
    layer_cpu.build((batch_size, timesteps, num_input))
    initial_weights = layer_cpu.get_weights()

    # Create CPU graph
    initial_state_cpu = keras.Input(batch_shape=(batch_size, num_hidden))
    inputs_cpu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
    outputs_cpu = layer_cpu(inputs_cpu, initial_state=initial_state_cpu)

    # Create, fit, and make prediction with CPU model
    model_cpu = keras.Model(inputs=(inputs_cpu, initial_state_cpu),
                            outputs=outputs_cpu)
    model_cpu.compile(loss='categorical_crossentropy', optimizer='adam')
    model_cpu.fit((x_fit, init), y, batch_size=batch_size)
    results_cpu = model_cpu.predict((x_predict, init), batch_size=batch_size)
    weights_cpu = layer_cpu.get_weights()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Setup IPU GRU layer
      layer_ipu = _getGRULayer(ipu.layers.PopnnGRU,
                               reset_after=True,
                               return_state=False)
      layer_ipu.build((batch_size, timesteps, num_input))
      layer_ipu.set_weights(initial_weights)

      # Create IPU graph
      initial_state_ipu = keras.Input(batch_shape=(batch_size, num_hidden))
      inputs_ipu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
      outputs_ipu = layer_ipu(inputs_ipu, initial_state=initial_state_ipu)

      # Create, fit, and make prediction with IPU model
      model_ipu = keras.Model(inputs=(inputs_ipu, initial_state_ipu),
                              outputs=outputs_ipu)
      model_ipu.compile(loss='categorical_crossentropy', optimizer='adam')
      model_ipu.fit((x_fit, init), y, batch_size=batch_size)
      results_ipu = model_ipu.predict((x_predict, init), batch_size=batch_size)
      weights_ipu = layer_ipu.get_weights()

    self.assertAllClose(results_ipu, results_cpu)
    self.assertAllClose(weights_ipu, weights_cpu)


if __name__ == '__main__':
  test.main()

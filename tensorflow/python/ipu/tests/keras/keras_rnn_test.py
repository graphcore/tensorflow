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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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
dataType = np.float32


def _tfLSTM(instance,
            x_val,
            h_val,
            c_val,
            return_sequences=False,
            time_major=False):
  with ops.device('cpu'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    h = array_ops.placeholder(h_val.dtype, h_val.shape)
    c = array_ops.placeholder(c_val.dtype, c_val.shape)
    lstm_cell = rnn_cell.LSTMCell(
        num_hidden,
        name='basic_lstm_cell',
        forget_bias=0.,
        initializer=init_ops.ones_initializer(dtype=dataType))
    state = rnn_cell.LSTMStateTuple(c, h)
    output = rnn.dynamic_rnn(lstm_cell,
                             x,
                             dtype=dataType,
                             initial_state=state,
                             time_major=time_major)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    out, outs = sess.run(output, {x: x_val, h: h_val, c: c_val})

    if not return_sequences:
      out = out[-1, :, :] if time_major else out[:, -1, :]

    return out, outs[1], outs[0]


def _new_kerasLSTM(instance,
                   x_val,
                   h_val,
                   c_val,
                   training=True,
                   return_state=True,
                   return_sequences=False,
                   time_major=False,
                   dropout=0.):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    h = array_ops.placeholder(h_val.dtype, h_val.shape)
    c = array_ops.placeholder(c_val.dtype, c_val.shape)
    state = rnn_cell.LSTMStateTuple(c, h)

    output = ipu.layers.PopnnLSTM(
        num_hidden,
        dtype=dataType,
        kernel_initializer=init_ops.ones_initializer(dtype=dataType),
        recurrent_initializer=init_ops.ones_initializer(dtype=dataType),
        bias_initializer=init_ops.ones_initializer(dtype=dataType),
        dropout=dropout,
        time_major=time_major,
        return_sequences=return_sequences,
        return_state=return_state)(inputs=x,
                                   initial_state=state,
                                   training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, h: h_val, c: c_val})


def _tfGRU(instance,
           x_val,
           initial_state_val,
           return_sequences=False,
           time_major=False):
  with ops.device('cpu'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    initial_state = array_ops.placeholder(initial_state_val.dtype,
                                          initial_state_val.shape)
    gru_cell = rnn_cell.GRUCell(
        num_hidden,
        name='gru_cell',
        kernel_initializer=init_ops.ones_initializer(dtype=dataType),
        bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType))
    output = rnn.dynamic_rnn(gru_cell,
                             x,
                             dtype=dataType,
                             initial_state=initial_state,
                             time_major=time_major)
  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    out, outs = sess.run(output, {x: x_val, initial_state: initial_state_val})

    if not return_sequences:
      out = out[-1, :, :] if time_major else out[:, -1, :]

    return out, outs


def _new_kerasGRU(instance,
                  x_val,
                  initial_state_val,
                  training=True,
                  return_state=True,
                  return_sequences=False,
                  time_major=False,
                  dropout=0.):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    initial_state = array_ops.placeholder(initial_state_val.dtype,
                                          initial_state_val.shape)
    output = ipu.layers.PopnnGRU(
        num_hidden,
        dtype=dataType,
        kernel_initializer=init_ops.ones_initializer(dtype=dataType),
        bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType),
        dropout=dropout,
        time_major=time_major,
        return_sequences=return_sequences,
        return_state=return_state)(inputs=x,
                                   initial_state=initial_state,
                                   training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, initial_state: initial_state_val})


class IpuLstmTest(test.TestCase):
  def _get_random_inputs(self, time_major=False):
    np.random.seed(42)
    h = np.random.rand(batch_size, num_hidden).astype(dataType)
    c = np.random.rand(batch_size, num_hidden).astype(dataType)
    if time_major:
      x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    else:
      x = np.random.rand(batch_size, timesteps, num_input).astype(dataType)
    return x, h, c

  @test_util.deprecated_graph_mode_only
  def test_lstm(self):
    x, h, c = self._get_random_inputs()

    keras_result = _new_kerasLSTM(self, x, h, c)
    result_tf = _tfLSTM(self, x, h, c)

    self.assertEqual(len(keras_result), 3)
    self.assertEqual(len(result_tf), 3)

    self.assertAllClose(keras_result[0], result_tf[0], rtol=0.01)
    self.assertAllClose(keras_result[1], result_tf[1], rtol=0.2)
    self.assertAllClose(keras_result[2], result_tf[2], rtol=0.2)

    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape, (batch_size, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_time_major(self):
    x, h, c = self._get_random_inputs(time_major=True)

    keras_result = _new_kerasLSTM(self, x, h, c, time_major=True)
    result_tf = _tfLSTM(self, x, h, c, time_major=True)

    self.assertEqual(len(keras_result), 3)
    self.assertEqual(len(result_tf), 3)

    self.assertAllClose(keras_result[0], result_tf[0], rtol=0.01)
    self.assertAllClose(keras_result[1], result_tf[1], rtol=0.2)
    self.assertAllClose(keras_result[2], result_tf[2], rtol=0.2)

    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape, (batch_size, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_all_seq(self):
    x, h, c = self._get_random_inputs()

    keras_result = _new_kerasLSTM(self, x, h, c, return_sequences=True)
    result_tf = _tfLSTM(self, x, h, c, return_sequences=True)

    self.assertEqual(len(keras_result), 3)
    self.assertEqual(len(result_tf), 3)

    self.assertAllClose(keras_result[0], result_tf[0], rtol=0.05)
    self.assertAllClose(keras_result[1], result_tf[1], rtol=0.2)
    self.assertAllClose(keras_result[2], result_tf[2], rtol=0.2)

    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_no_state(self):
    x, h, c = self._get_random_inputs()

    keras_result = _new_kerasLSTM(self, x, h, c, return_state=False)
    self.assertTrue(isinstance(keras_result, np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_no_dynamic_training(self):
    x, h, c = self._get_random_inputs()

    with self.assertRaisesRegex(ValueError,
                                'PopnnLSTM does not support a dynamic'):
      _new_kerasLSTM(self, x, h, c, training=None)

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(ipu.layers.LSTM, type))
    self.assertEqual(ipu.layers.PopnnLSTM, ipu.layers.LSTM)

  @test_util.deprecated_graph_mode_only
  def test_lstm_dropout(self):
    x, h, c = self._get_random_inputs()

    dropout_result = _new_kerasLSTM(self,
                                    x,
                                    h,
                                    c,
                                    return_state=False,
                                    dropout=0.)

    clear_result = _new_kerasLSTM(self,
                                  x,
                                  h,
                                  c,
                                  return_state=False,
                                  dropout=1.)

    self.assertNotAllClose(clear_result, dropout_result)

  @test_util.run_v2_only
  def test_can_call_without_state_change(self):
    x, h, c = self._get_random_inputs()

    layer = ipu.layers.PopnnLSTM(
        num_hidden,
        dtype=dataType,
        kernel_initializer=init_ops.random_uniform_initializer(seed=42,
                                                               dtype=dataType),
        recurrent_initializer=init_ops.random_uniform_initializer(
            seed=42, dtype=dataType),
        bias_initializer=init_ops.zeros_initializer(dtype=dataType))
    layer.build(x.shape)

    @def_function.function
    def impl(x, c, h):
      state = rnn_cell.LSTMStateTuple(c, h)
      return layer(inputs=x, initial_state=state)

    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x, c, h)
    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x, c, h)


class IpuGruTest(test.TestCase):
  def _get_random_inputs(self, time_major=False):
    np.random.seed(42)
    init = np.random.rand(batch_size, num_hidden).astype(dataType)
    if time_major:
      x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    else:
      x = np.random.rand(batch_size, timesteps, num_input).astype(dataType)
    return x, init

  @test_util.deprecated_graph_mode_only
  def test_gru(self):
    x, init = self._get_random_inputs()

    keras_result = _new_kerasGRU(self, x, init)
    result_tf = _tfGRU(self, x, init)

    self.assertAllClose(keras_result, result_tf)
    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape, (batch_size, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_gru_seq_major(self):
    x, init = self._get_random_inputs(True)

    keras_result = _new_kerasGRU(self, x, init, time_major=True)
    result_tf = _tfGRU(self, x, init, time_major=True)

    self.assertAllClose(keras_result, result_tf)
    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape, (batch_size, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_gru_all_seq(self):
    x, init = self._get_random_inputs()

    keras_result = _new_kerasGRU(self, x, init, return_sequences=True)
    result_tf = _tfGRU(self, x, init, return_sequences=True)

    self.assertAllClose(keras_result, result_tf)
    self.assertNotAllClose(keras_result[0], np.ones([batch_size, num_hidden]))
    self.assertEqual(keras_result[0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_gru_no_state(self):
    x, init = self._get_random_inputs()

    keras_result = _new_kerasGRU(self, x, init, return_state=False)
    self.assertTrue(isinstance(keras_result, np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_no_dynamic_training(self):
    x, init = self._get_random_inputs()

    with self.assertRaisesRegex(ValueError,
                                'PopnnGRU does not support a dynamic'):
      _new_kerasGRU(self, x, init, training=None)

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(ipu.layers.GRU, type))
    self.assertEqual(ipu.layers.PopnnGRU, ipu.layers.GRU)

  @test_util.deprecated_graph_mode_only
  def test_gru_dropout(self):
    x, init = self._get_random_inputs()

    dropout_result = _new_kerasGRU(self,
                                   x,
                                   init,
                                   dropout=1.,
                                   return_state=False,
                                   return_sequences=True)
    clear_result = _new_kerasGRU(self,
                                 x,
                                 init,
                                 dropout=0.,
                                 return_state=False,
                                 return_sequences=True)

    self.assertNotAllClose(clear_result, dropout_result)


if __name__ == '__main__':
  test.main()

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

from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables

from tensorflow.python import ipu

# Test hyperparameters.
batch_size = 8
num_input = 2
timesteps = 4
num_hidden = 18
dataType = np.float32


def _tfLSTM(instance, x_val, h_val, c_val):
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
                             time_major=True)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, h: h_val, c: c_val})


def _new_kerasLSTM(instance,
                   x_val,
                   h_val,
                   c_val,
                   training=True,
                   return_state=True):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    h = array_ops.placeholder(h_val.dtype, h_val.shape)
    c = array_ops.placeholder(c_val.dtype, c_val.shape)
    state = rnn_cell.LSTMStateTuple(c, h)
    output = ipu.layers.PopnnLSTM(
        num_hidden,
        dtype=dataType,
        weights_initializer=init_ops.ones_initializer(dtype=dataType),
        recurrent_weight_initializer=None,
        bias_initializer=init_ops.zeros_initializer(dtype=dataType),
        return_state=return_state)(inputs=x,
                                   initial_state=state,
                                   training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, h: h_val, c: c_val})


def _tfGRU(instance, x_val, initial_state_val):
  with ops.device('cpu'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    initial_state = array_ops.placeholder(initial_state_val.dtype,
                                          initial_state_val.shape)

    gru_cell = rnn_cell.GRUCell(
        num_hidden,
        name='gru_cell',
        kernel_initializer=init_ops.zeros_initializer(dtype=dataType),
        bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType))
    output = rnn.dynamic_rnn(gru_cell,
                             x,
                             dtype=dataType,
                             initial_state=initial_state,
                             time_major=True)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, initial_state: initial_state_val})


def _new_kerasGRU(instance,
                  x_val,
                  initial_state_val,
                  training=True,
                  return_state=True):
  with ops.device('/device:IPU:0'):
    x = array_ops.placeholder(x_val.dtype, x_val.shape)
    initial_state = array_ops.placeholder(initial_state_val.dtype,
                                          initial_state_val.shape)
    output = ipu.layers.PopnnGRU(
        num_hidden,
        dtype=dataType,
        weights_initializer=init_ops.zeros_initializer(dtype=dataType),
        bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType),
        return_state=return_state)(inputs=x,
                                   initial_state=initial_state,
                                   training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {x: x_val, initial_state: initial_state_val})


class IpuLstmTest(test.TestCase):
  @test_util.deprecated_graph_mode_only
  def testLstm(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)

    np.random.seed(42)
    keras_result = _new_kerasLSTM(
        self, x, np.ones((batch_size, num_hidden), dtype=dataType),
        np.ones((batch_size, num_hidden), dtype=dataType))

    np.random.seed(42)
    result_tf = _tfLSTM(self, x,
                        np.ones((batch_size, num_hidden), dtype=dataType),
                        np.ones((batch_size, num_hidden), dtype=dataType))
    self.assertAllClose(keras_result, result_tf)

  def test_lstm_no_state(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)

    np.random.seed(42)
    keras_result = _new_kerasLSTM(self,
                                  x,
                                  np.ones((batch_size, num_hidden),
                                          dtype=dataType),
                                  np.ones((batch_size, num_hidden),
                                          dtype=dataType),
                                  training=True,
                                  return_state=False)
    self.assertTrue(isinstance(keras_result, np.ndarray))

  def test_no_dynamic_training(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    # Get the "normal" tensorflow result
    with self.assertRaisesRegex(ValueError,
                                'PopnnLSTM does not support a dynamic'):
      _new_kerasLSTM(self,
                     x,
                     np.ones((batch_size, num_hidden), dtype=dataType),
                     array_ops.ones((batch_size, num_hidden), dtype=dataType),
                     training=None)


class IpuGruTest(test.TestCase):
  @test_util.deprecated_graph_mode_only
  def testGru(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    # Get the "normal" tensorflow result
    keras_result = _new_kerasGRU(
        self, x, np.ones((batch_size, num_hidden), dtype=dataType))
    result_tf = _tfGRU(self, x,
                       np.ones((batch_size, num_hidden), dtype=dataType))
    # Check they are the same.
    self.assertAllClose(keras_result, result_tf)

  def test_lstm_no_state(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)

    np.random.seed(42)
    keras_result = _new_kerasGRU(self,
                                 x,
                                 np.ones((batch_size, num_hidden),
                                         dtype=dataType),
                                 return_state=False)
    self.assertTrue(isinstance(keras_result, np.ndarray))

  def test_no_dynamic_training(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    # Get the "normal" tensorflow result
    with self.assertRaisesRegex(ValueError,
                                'PopnnGRU does not support a dynamic'):
      _new_kerasGRU(self,
                    x,
                    np.ones((batch_size, num_hidden), dtype=dataType),
                    training=None)


if __name__ == '__main__':
  test.main()

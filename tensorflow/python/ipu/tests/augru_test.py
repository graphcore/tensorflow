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
# ==============================================================================
"""Tests covering augru used by the DIEN model."""

from functools import partial
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.client import session
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ipu import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ops.rnn_ops import PopnnAUGRU
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class TestDIENAUGRU(test_util.TensorFlowTestCase):
  """Testing augru layer"""
  @classmethod
  def setUpClass(cls):
    cls.model_dtype = dtypes.float32
    cls.HIDDEN_SIZE = 2

  def test_augru(self):
    seq_len = 2
    bs = 3
    inputs_value = np.ones([seq_len, bs, self.HIDDEN_SIZE], np.float32)
    seq_len_value = np.array([1, 2, 2], np.int32)
    alphas_value = np.ones([seq_len, bs], np.float32)
    alphas_value = alphas_value * 0.5
    inputs_ph = array_ops.placeholder(shape=[seq_len, bs, self.HIDDEN_SIZE],
                                      dtype=self.model_dtype)
    seq_len_ph = array_ops.placeholder(shape=[bs], dtype=dtypes.int32)
    alphas_ph = array_ops.placeholder(shape=[seq_len, bs],
                                      dtype=self.model_dtype)
    gru_kernel = np.zeros((4, 6))
    cfg = IPUConfig()
    cfg._profiling.profiling = False  # pylint: disable=protected-access
    cfg._profiling.profile_execution = False  # pylint: disable=protected-access
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()
    with ops.device("/device:IPU:0"):
      time_major_model = partial(self.augru_model,
                                 time_major=True,
                                 scope_name="time_major")
      batch_major_model = partial(self.augru_model,
                                  time_major=False,
                                  scope_name="batch_major")
      train_ipu_time_major = ipu_compiler.compile(
          time_major_model, inputs=[inputs_ph, seq_len_ph, alphas_ph])
      train_ipu_batch_major = ipu_compiler.compile(
          batch_major_model, inputs=[inputs_ph, seq_len_ph, alphas_ph])
      train_cpu = ipu_compiler.compile(
          self.augru_model_cpu, inputs=[inputs_ph, seq_len_ph, alphas_ph])

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      for var in variables.global_variables():
        if var.name == 'time_major/popnn_augru/biases:0' \
            or var.name == 'batch_major/popnn_augru/biases:0':
          gru_bias = np.array([[1, 1], [1, 1], [0, 0]])
          gru_bias_var = var
          sess.run(state_ops.assign(gru_bias_var, gru_bias))
      for var in variables.global_variables():
        if var.name == 'time_major/popnn_augru/kernel:0' \
            or var.name == 'batch_major/popnn_augru/kernel:0':
          gru_kernel_var = var
        if var.name == 'augru2/vec_att_gru_cell/gates/kernel:0':
          gru_u_r_kernel = sess.run(var)
        if var.name == 'augru2/vec_att_gru_cell/candidate/kernel:0':
          gru_c_kernel = sess.run(var)

      # need to roll here because the cellorder is different
      # from the implementation of the ipu version
      gru_kernel = array_ops.concat(
          [np.roll(gru_u_r_kernel, 2, axis=1), gru_c_kernel], 1)
      for var in variables.global_variables():
        if var.name == 'time_major/popnn_augru/kernel:0' \
            or var.name == 'batch_major/popnn_augru/kernel:0':
          sess.run(tf.assign(var, gru_kernel))

      outputs_expected = np.array(
          sess.run(train_cpu,
                   feed_dict={
                       inputs_ph: inputs_value,
                       seq_len_ph: seq_len_value,
                       alphas_ph: alphas_value
                   }))
      outputs_time_major = np.array(
          sess.run(train_ipu_time_major,
                   feed_dict={
                       inputs_ph: inputs_value,
                       seq_len_ph: seq_len_value,
                       alphas_ph: alphas_value
                   }))
      outputs_batch_major = np.array(
          sess.run(train_ipu_batch_major,
                   feed_dict={
                       inputs_ph: inputs_value,
                       seq_len_ph: seq_len_value,
                       alphas_ph: alphas_value
                   }))

      gru_kernel_updated = sess.run(gru_kernel_var)

      # get the updated weights
      for var in variables.global_variables():
        if var.name == 'augru2/vec_att_gru_cell/gates/kernel:0':
          gru_kernel_u_r_cpu = sess.run(var)
        if var.name == 'augru2/vec_att_gru_cell/candidate/kernel:0':
          gru_kernel_c_cpu = sess.run(var)
      cpu_kernel_u_r = gru_kernel_u_r_cpu
      cpu_kernel_c = gru_kernel_c_cpu
      # need to roll again to match the input
      kernel_expeted = np.concatenate(
          [np.roll(cpu_kernel_u_r, 2, axis=1), cpu_kernel_c], axis=1)

      self.assertAlmostEqual(np.mean(outputs_time_major - outputs_expected),
                             np.float32(0.0),
                             delta=1e-7)
      self.assertAlmostEqual(np.mean(outputs_batch_major - outputs_expected),
                             np.float32(0.0),
                             delta=1e-7)
      self.assertAlmostEqual(np.mean(gru_kernel_updated - kernel_expeted),
                             np.float32(0.0),
                             delta=1e-8)

  def augru_model(self, inputs, seq_len, alphas, time_major, scope_name):
    with variable_scope.variable_scope(scope_name):
      augru = PopnnAUGRU(self.HIDDEN_SIZE)
      if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])
        alphas = array_ops.transpose(alphas, [1, 0])
      rnn_outputs, _ = augru(inputs, seq_len, alphas, time_major=time_major)
      rnn_outputs = array_ops.transpose(rnn_outputs, [1, 0, 2])
      loss = math_ops.reduce_mean(rnn_outputs - 1.0)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1)
      grads_and_vars = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(grads_and_vars)
      return rnn_outputs, train_op

  def augru_model_cpu(self, inputs, seq_len, alphas):
    inputs = tf.transpose(inputs, [1, 0, 2])
    alphas = array_ops.reshape(alphas, [array_ops.shape(inputs)[0], -1])
    alphas = array_ops.expand_dims(alphas, -1)
    # concat attention in front of inputs
    inputs = array_ops.concat([inputs, alphas], axis=2)
    rnn_outputs2, _ = dynamic_rnn(VecAttGRUCell(self.HIDDEN_SIZE),
                                  inputs=inputs,
                                  sequence_length=seq_len,
                                  dtype=self.model_dtype,
                                  scope="augru2")
    loss = math_ops.reduce_mean(rnn_outputs2 - 1.0)
    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    return rnn_outputs2, train_op


# implementation of the cpu version of augru
class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """
  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # [batch, seq_len, input_state]
    # The attention infomation is after the end of inputs data.
    seq_len = inputs.shape[0]
    hidden_size = inputs.shape[1]
    inputs_begin = array_ops.constant(np.array([0, 0]), dtype=dtypes.int32)
    inputs_end = array_ops.constant(np.array([seq_len, hidden_size - 1]),
                                    dtype=dtypes.int32)

    attention_begin = array_ops.constant(np.array([0, hidden_size - 1]),
                                         dtype=dtypes.int32)
    attention_end = array_ops.constant(np.array([seq_len, 1]),
                                       dtype=dtypes.int32)
    real_inputs = array_ops.slice(inputs, inputs_begin, inputs_end)
    att_score = array_ops.slice(inputs, attention_begin, attention_end)

    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with variable_scope.variable_scope(
          "gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [real_inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([real_inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with variable_scope.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [real_inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([real_inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """
  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]

    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError(
            "linear expects shape[1] to be provided for shape %s, "
            "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = variable_scope.get_variable_scope()
    with variable_scope.variable_scope(scope) as outer_scope:
      self._weights = variable_scope.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with variable_scope.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = variable_scope.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Naive GRU to learn three-char time steps to one-char mapping

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.config import IPUConfig

dataType = np.float32

seq_len = 3
batch_size = 40 - seq_len
input_size = 1
num_hidden = 64
num_training_steps = 100
lr = 10


# pylint: disable=unused-argument
def _PopnnGRU(x, initial_state, y, sequence_len=None):
  gru_cell = ipu.ops.rnn_ops.PopnnGRU(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType),
      reset_after=False)
  outputs, _ = gru_cell(x, initial_state=initial_state, training=True)
  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


# pylint: disable=unused-argument
def _PopnnGRU_DynamicGRU(x, initial_state, y, sequence_len=None):
  gru_cell = ipu.ops.rnn_ops.PopnnDynamicGRU(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType),
      reset_after=False)
  outputs, _ = gru_cell(x,
                        sequence_len,
                        initial_state=initial_state,
                        training=True)

  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


# pylint: disable=unused-argument
def _PopnnGRU_ResetAfter(x, initial_state, y, sequence_len=None):
  gru_cell = ipu.ops.rnn_ops.PopnnGRU(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType),
      reset_after=True)
  outputs, _ = gru_cell(x, initial_state=initial_state, training=True)
  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


def _tfGRU(x, initial_state, y, sequence_len=None):
  gru_cell = rnn_cell.GRUCell(
      num_hidden,
      name='gru_cell',
      kernel_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType))
  outputs, _ = rnn.dynamic_rnn(gru_cell,
                               x,
                               sequence_length=sequence_len,
                               dtype=dataType,
                               initial_state=initial_state,
                               time_major=True)

  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


def get_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class GRUTrainingTest(xla_test.XLATestCase):
  def _RunLayer(self, layer_func, x, y, s=None):
    with self.session() as sess:
      with ops.device('cpu'):
        px = array_ops.placeholder(dataType, shape=x.shape)
        pi_state = array_ops.placeholder(dataType,
                                         shape=[batch_size, num_hidden])
        py = array_ops.placeholder(dataType, shape=y.shape)
        compile_inputs = [px, pi_state, py]
        fd = {px: x, pi_state: np.zeros(pi_state.shape), py: y}

        if s is not None:
          ps = array_ops.placeholder(np.int32, shape=s.shape)
          compile_inputs.append(ps)
          fd[ps] = s

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(layer_func, inputs=compile_inputs)

      opts = IPUConfig()
      opts._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = True  # pylint: disable=protected-access
      opts.ipu_model.compile_ipu_code = False
      opts.configure_ipu_system()

      sess.run(variables.global_variables_initializer())
      losses = []
      for _ in range(0, num_training_steps):
        loss = sess.run(r, fd)
        losses.append(loss)
    return losses

  # Check that the loss goes down (and is identical to reference version).
  def testTraining(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      inputs.append(sequence)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    # normalize
    X = X / float(len(nums))

    # geneate a target
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:, 0] = 1.

    custom_losses = self._RunLayer(_PopnnGRU, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._RunLayer(_tfGRU, X, labels)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)

  def testTrainingWithSeqLen(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      inputs.append(sequence)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    S = np.array([(i % seq_len) + 1 for i in range(batch_size)])
    # normalize
    X = X / float(len(nums))

    # Generate a target
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:, 0] = 1.

    custom_losses = self._RunLayer(_PopnnGRU_DynamicGRU, X, labels, s=S)

    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._RunLayer(_tfGRU, X, labels, s=S)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)

  def testTraining_resetAfter(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      inputs.append(sequence)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    # normalize
    X = X / float(len(nums))

    # generate a target
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:, 0] = 1.

    custom_losses = self._RunLayer(_PopnnGRU_ResetAfter, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])

    # TF GRU does not support reset_after so no reference comparison
    # is done here.


if __name__ == "__main__":
  googletest.main()

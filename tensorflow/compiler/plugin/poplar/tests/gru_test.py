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
"""Tests for the GRU cell and layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

# pylint: disable=unused-import
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import rnn_ops_grad
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
# pylint: enable=unused-import

dataType = np.float32
batch_size = 1
seq_len = 3
input_size = 5
num_channels = 8


def _get_variable(name, shape, initializer):
  return variable_scope.get_variable(
      name, shape=shape, initializer=initializer, dtype=dataType)


def _createGRUInput(value, batch_size, seq_len, input_size):
  return np.full(
      fill_value=value,
      shape=[seq_len, batch_size, input_size],
      dtype=dataType)


def _createGRUInitialState(value, batch_size, num_channels):
  return np.full(
      fill_value=value, shape=[batch_size, num_channels], dtype=dataType)


class GRUTest(xla_test.XLATestCase):
  def _GRULayerCPU(self, inputs, weights_value, initial_state, training, name):
    with ops.device("/device:CPU:0"):
      gru_cell = rnn_cell.GRUCell(
          num_channels,
          name='gru_cell',
          kernel_initializer=init_ops.constant_initializer(
              weights_value, dtype=dataType),
          bias_initializer=init_ops.zeros_initializer(dtype=dataType),
          reuse=variable_scope.AUTO_REUSE)

      outputs, state = rnn.dynamic_rnn(
          gru_cell,
          inputs,
          dtype=dataType,
          initial_state=initial_state,
          time_major=True)
      return outputs

  def _GRULayer(self, inputs, weights_value, initial_state, training, name):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("gru_layer", use_resource=True):
        kernel = _get_variable(
            "kernel",
            shape=[input_size + num_channels, 3 * num_channels],
            initializer=init_ops.constant_initializer(weights_value, dataType))
        biases = _get_variable(
            "biases",
            shape=[3, num_channels],
            initializer=init_ops.constant_initializer(0.0, dataType))
      outputs, _, _ = gen_popnn_ops.popnn_gru_layer(
          inputs=inputs,
          num_channels=num_channels,
          kernel=kernel,
          biases=biases,
          initial_state=initial_state,
          is_training=training,
          name=name)
      return outputs

  def _RunGRULayerInference(self, name, input_value, weights_value,
                            init_state_value, gru_layer_function):
    with self.session() as sess:
      pinputs = array_ops.placeholder(
          dataType, [seq_len, batch_size, input_size], name="inputs")
      pinitial_state = array_ops.placeholder(
          dataType, [batch_size, num_channels], name="initial_state")
      gru_output_seq = gru_layer_function(
          inputs=pinputs,
          weights_value=weights_value,
          initial_state=pinitial_state,
          training=False,
          name=name)

      inputs = _createGRUInput(input_value, batch_size, seq_len, input_size)
      initial_state = _createGRUInitialState(init_state_value, batch_size,
                                             num_channels)
      fd = {
          pinputs: inputs,
          pinitial_state: initial_state,
      }
      sess.run(variables.global_variables_initializer())
      return sess.run(gru_output_seq, fd)

  def _RunInferenceComparison(self, name, input_value, weights_value,
                              init_state_value):
    ops.reset_default_graph()
    popnn_out = self._RunGRULayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        init_state_value=init_state_value,
        gru_layer_function=self._GRULayer)
    ref_out = self._RunGRULayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        init_state_value=init_state_value,
        gru_layer_function=self._GRULayerCPU)
    # Check that the whole outupt sequence matches
    self.assertAllClose(popnn_out, ref_out)

  def testGRULayerInference(self):
    tu.configure_ipu_system(True, True, True)
    np.random.seed(0)
    # Run with all-0 weights
    weight0 = 0.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison(
          'ones',
          input_value=0.,
          weights_value=weight0,
          init_state_value=init_state_value)

    # Run with all-1 weights
    weight1 = 1.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison(
          'ones',
          input_value=0.,
          weights_value=weight1,
          init_state_value=init_state_value)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunInferenceComparison(
            'rand',
            input_value=0.,
            weights_value=weight,
            init_state_value=init_state_value)

  def _RunGRULayerTraining(self, name, input_value, weights_value,
                           init_state_value, training_steps, labels_array,
                           gru_layer_function, device_string):
    with self.session() as sess:
      pinputs = array_ops.placeholder(
          dataType, [seq_len, batch_size, input_size], name="inputs")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      with ops.device(device_string):
        with variable_scope.variable_scope("gru_layer", use_resource=True):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(
                  init_state_value, dataType))
        logits = gru_layer_function(
            inputs=pinputs,
            weights_value=weights_value,
            initial_state=initial_state,
            training=True,
            name=name)
        logits = math_ops.reduce_mean(logits, axis=0)
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      losses = []
      inputs = _createGRUInput(input_value, batch_size, seq_len, input_size)
      fd = {
          pinputs: inputs,
          plabels: labels_array,
      }
      for _ in range(0, training_steps):
        l, _ = sess.run([loss, train], fd)
        losses.append(l)
      return losses

  def _RunTrainingComparison(self, name, input_value, weights_value,
                             init_state_value, training_steps):
    labels_array = np.ones(shape=[batch_size], dtype=np.int32)
    ops.reset_default_graph()
    popnn_losses = self._RunGRULayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        init_state_value=init_state_value,
        training_steps=training_steps,
        labels_array=labels_array,
        gru_layer_function=self._GRULayer,
        device_string="/device:IPU:0")
    ops.reset_default_graph()
    ref_losses = self._RunGRULayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        init_state_value=init_state_value,
        training_steps=training_steps,
        labels_array=labels_array,
        gru_layer_function=self._GRULayerCPU,
        device_string="/device:CPU:0")
    self.assertAllClose(popnn_losses, ref_losses)

  def testGRULayerTraining(self):
    tu.configure_ipu_system(True, True, True)
    np.random.seed(42)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison(
            'rand',
            input_value=0.,
            weights_value=weight,
            init_state_value=init_state_value,
            training_steps=3)

  def testGRUCached(self):
    with self.session() as sess:
      pinputs1 = array_ops.placeholder(
          dataType, [seq_len, batch_size, input_size], name="inputs1")
      pinputs2 = array_ops.placeholder(
          dataType, [seq_len, batch_size, input_size], name="inputs2")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      with ops.device("/device:IPU:0"):

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(0.1, dataType))
          return self._GRULayer(
              inputs=inputs,
              weights_value=1.,
              initial_state=initial_state,
              training=True,
              name=name)

        with variable_scope.variable_scope("gru_layer1", use_resource=True):
          logits1 = gru_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("gru_layer2", use_resource=True):
          logits2 = gru_layer(pinputs2, "layer2")

        logits = (math_ops.reduce_mean(logits1, axis=0) + math_ops.reduce_mean(
            logits2, axis=0))
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)
      sess.run(
          [loss, train], {
              pinputs1: _createGRUInput(0.5, batch_size, seq_len, input_size),
              pinputs2: _createGRUInput(1.5, batch_size, seq_len, input_size),
              plabels: np.ones(shape=[batch_size], dtype=np.int32),
          })

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Check there is one fwd GRU
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list,
                                         '*/CalcNextOutput/Op/Multiply'), 1)
      # Check there is one bwd GRU
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list, '*/MulOGate/Op/Multiply'), 1)

  def testGRUNotCached(self):
    with self.session() as sess:
      # Note here the second GRU is larger.
      pinputs1 = array_ops.placeholder(
          dataType, [seq_len, batch_size, input_size], name="inputs1")
      pinputs2 = array_ops.placeholder(
          dataType, [seq_len * 2, batch_size, input_size], name="inputs2")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      with ops.device("/device:IPU:0"):

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(0.1, dataType))
          return self._GRULayer(
              inputs=inputs,
              weights_value=1.,
              initial_state=initial_state,
              training=True,
              name=name)

        with variable_scope.variable_scope("gru_layer1", use_resource=True):
          logits1 = gru_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("gru_layer2", use_resource=True):
          logits2 = gru_layer(pinputs2, "layer2")

        logits = (math_ops.reduce_mean(logits1, axis=0) + math_ops.reduce_mean(
            logits2, axis=0))
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)
      sess.run(
          [loss, train], {
              pinputs1: _createGRUInput(0.5, batch_size, seq_len, input_size),
              pinputs2: _createGRUInput(1.5, batch_size, seq_len * 2,
                                        input_size),
              plabels: np.ones(shape=[batch_size], dtype=np.int32),
          })

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # Check there are two fwd GRUs.
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list,
                                         '*/CalcNextOutput/Op/Multiply'), 2)
      # Check there are two bwd GRUs.
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list, '*/MulOGate/Op/Multiply'), 2)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

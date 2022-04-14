# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the LSTM cell and layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl.testing import parameterized
import numpy as np
import pva
from tensorflow.python.ipu import test_utils as tu

# pylint: disable=unused-import
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.keras.layers import LSTM
# pylint: enable=unused-import

DATA_TYPE = np.float32
BATCH_SIZE = 1
SEQ_LEN = 3
INPUT_SIZE = 3
NUM_CHANNELS = 4


def _get_variable(name, shape, initializer):
  return variable_scope.get_variable(name,
                                     shape=shape,
                                     initializer=initializer,
                                     dtype=DATA_TYPE)


def _createLSTMInput(value, batch_size, seq_len, input_size):
  return np.full(fill_value=value,
                 shape=[seq_len, batch_size, input_size],
                 dtype=DATA_TYPE)


def _createLSTMInitialState(h_value, c_value, batch_size, num_channels):
  return (np.full(fill_value=h_value,
                  shape=[batch_size, num_channels],
                  dtype=DATA_TYPE),
          np.full(fill_value=c_value,
                  shape=[batch_size, num_channels],
                  dtype=DATA_TYPE))


def _layerInferenceTestCases():
  cases = []
  for h_init in [0., 1.]:
    for c_init in [0., 1.]:
      cases.append({
          'testcase_name': 'h_init_%f_c_init%f' % (h_init, c_init),
          'h_init': h_init,
          'c_init': c_init
      })
  return cases


def _activationTestCases():
  cases = []
  for activation in ['tanh', 'relu', 'softmax', 'sigmoid', 'hard_sigmoid']:
    for recurrent_activation in ['softmax', 'sigmoid', 'hard_sigmoid']:
      cases.append({
          'testcase_name':
          '%s_%s' % (activation, recurrent_activation),
          'activation':
          activation,
          'recurrent_activation':
          recurrent_activation
      })
  return cases


def _totalTileMemory(report):
  return sum(tile.memory.total.excludingGaps
             for tile in report.compilation.tiles)


LAYER_WEIGHT_CASES = _layerInferenceTestCases()


class LSTMTest(xla_test.XLATestCase, parameterized.TestCase):  # pylint: disable=W0223
  def _LSTMLayerCPU(
      self,
      inputs,
      weights_value,
      initial_state,
      forget_bias,
      training,
      name,
      seq_lens=None,  # pylint: disable=unused-argument
      seq_lens_h=None,
      activation='tanh',
      recurrent_activation='sigmoid',
      input_size=INPUT_SIZE,  # pylint: disable=unused-argument
      num_channels=NUM_CHANNELS):
    del forget_bias
    del name
    with ops.device("/device:CPU:0"):
      lstm = LSTM(num_channels,
                  activation=activation,
                  recurrent_activation=recurrent_activation,
                  kernel_initializer=init_ops.constant_initializer(
                      weights_value, DATA_TYPE),
                  recurrent_initializer=init_ops.constant_initializer(
                      weights_value, DATA_TYPE),
                  bias_initializer=init_ops.constant_initializer(
                      0.0, DATA_TYPE),
                  time_major=True,
                  return_sequences=True,
                  stateful=True,
                  unit_forget_bias=False)
      outputs = lstm(inputs, initial_state=initial_state, training=training)
      outputs = outputs if seq_lens_h is None else outputs[0:min(
          SEQ_LEN, seq_lens_h[0])]
      return outputs

  def _LSTMLayer(self,
                 inputs,
                 weights_value,
                 initial_state,
                 forget_bias,
                 training,
                 name,
                 seq_lens=None,
                 seq_lens_h=None,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 input_size=INPUT_SIZE,
                 num_channels=NUM_CHANNELS,
                 options=None,
                 options_bwd=None):
    del forget_bias
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("lstm_layer", use_resource=True):
        kernel = _get_variable(
            "kernel",
            shape=[input_size + num_channels, 4 * num_channels],
            initializer=init_ops.constant_initializer(weights_value,
                                                      DATA_TYPE))
        biases = _get_variable("biases",
                               shape=[4, num_channels],
                               initializer=init_ops.constant_initializer(
                                   0.0, DATA_TYPE))
      options = '{}' if options is None else json.dumps(options)
      options_bwd = '{}' if options is None else json.dumps(options_bwd)
      if seq_lens is None:
        outputs, _, _, _ = gen_popnn_ops.popnn_lstm_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            input_h_state=initial_state[0],
            input_c_state=initial_state[1],
            is_training=training,
            name=name,
            options=options,
            options_bwd=options_bwd)
      else:
        outputs, _, _, _ = gen_popnn_ops.popnn_dynamic_lstm_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            seq_len=seq_lens,
            input_h_state=initial_state[0],
            input_c_state=initial_state[1],
            is_training=training,
            name=name,
            options=options,
            options_bwd=options_bwd)
      outputs = outputs if seq_lens_h is None else outputs[0:min(
          SEQ_LEN, seq_lens_h[0])]
      return outputs

  def _RunLSTMLayerInference(self, name, input_value, forget_bias,
                             weights_value, h_value, c_value, seq_lens,
                             lstm_layer_function):
    with self.session() as sess:
      pinputs = array_ops.placeholder(DATA_TYPE,
                                      [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                      name="inputs")
      pinitial_h_state = array_ops.placeholder(DATA_TYPE,
                                               [BATCH_SIZE, NUM_CHANNELS],
                                               name="init_h_state")
      pinitial_c_state = array_ops.placeholder(DATA_TYPE,
                                               [BATCH_SIZE, NUM_CHANNELS],
                                               name="init_c_state")
      pseq_lens = array_ops.placeholder(
          np.int32, [BATCH_SIZE],
          name="seq_len") if seq_lens is not None else None

      lstm_output_seq = lstm_layer_function(inputs=pinputs,
                                            weights_value=weights_value,
                                            initial_state=(pinitial_h_state,
                                                           pinitial_c_state),
                                            forget_bias=forget_bias,
                                            seq_lens=pseq_lens,
                                            seq_lens_h=seq_lens,
                                            training=False,
                                            name=name)

      inputs = _createLSTMInput(input_value, BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
      initial_state = _createLSTMInitialState(h_value, c_value, BATCH_SIZE,
                                              NUM_CHANNELS)
      fd = {
          pinputs: inputs,
          pinitial_h_state: initial_state[0],
          pinitial_c_state: initial_state[1],
      }
      if seq_lens is not None:
        fd[pseq_lens] = seq_lens

      sess.run(variables.global_variables_initializer())
      return sess.run(lstm_output_seq, fd)

  def _RunInferenceComparison(self,
                              name,
                              input_value,
                              forget_bias,
                              weights_value,
                              h_value,
                              c_value,
                              seq_lens=None):
    ops.reset_default_graph()
    popnn_out = self._RunLSTMLayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        seq_lens=seq_lens,
        lstm_layer_function=self._LSTMLayer)
    ref_out = self._RunLSTMLayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        seq_lens=seq_lens,
        lstm_layer_function=self._LSTMLayerCPU)
    # Check that the whole outupt sequence matches
    self.assertAllClose(popnn_out, ref_out)

  @parameterized.named_parameters(*LAYER_WEIGHT_CASES)
  def testLSTMLayerInferenceAllZeroWeight(self, h_init, c_init):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(0)
    # Run with all-0 weights
    weight0 = 1.
    self._RunInferenceComparison('ones',
                                 input_value=0.,
                                 forget_bias=0.,
                                 weights_value=weight0,
                                 h_value=h_init,
                                 c_value=c_init)

  @parameterized.named_parameters(*LAYER_WEIGHT_CASES)
  def testLSTMLayerInferenceAllOneWeight(self, h_init, c_init):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(0)
    # Run with all-1 weights
    weight1 = 1.
    self._RunInferenceComparison('ones',
                                 input_value=0.,
                                 forget_bias=0.,
                                 weights_value=weight1,
                                 h_value=h_init,
                                 c_value=c_init)

  @parameterized.named_parameters(*LAYER_WEIGHT_CASES)
  def testLSTMLayerInferenceRandomWeight(self, h_init, c_init):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(0)

    # Run with random weights
    for weight in np.random.rand(3):
      self._RunInferenceComparison('rand',
                                   input_value=0.,
                                   forget_bias=0.,
                                   weights_value=weight,
                                   h_value=h_init,
                                   c_value=c_init)

    # seq_len 1
    for weight in np.random.rand(3):
      self._RunInferenceComparison('rand',
                                   input_value=0.,
                                   forget_bias=0.,
                                   weights_value=weight,
                                   h_value=h_init,
                                   c_value=c_init,
                                   seq_lens=[1])

  def _RunLSTMLayerTraining(self,
                            name,
                            input_value,
                            forget_bias,
                            weights_value,
                            h_value,
                            c_value,
                            training_steps,
                            seq_lens,
                            labels_array,
                            lstm_layer_function,
                            device_string,
                            batch_size=BATCH_SIZE,
                            input_size=INPUT_SIZE,
                            num_channels=NUM_CHANNELS,
                            options=None,
                            options_bwd=None):
    with self.session() as sess:
      pinputs = array_ops.placeholder(DATA_TYPE,
                                      [SEQ_LEN, batch_size, input_size],
                                      name="inputs")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      pseq_lens = array_ops.placeholder(
          np.int32, [batch_size],
          name="seq_len") if seq_lens is not None else None

      with ops.device(device_string):
        with variable_scope.variable_scope("lstm_layer", use_resource=True):
          initial_h_state = _get_variable(
              "initial_h_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(h_value, DATA_TYPE))
          initial_c_state = _get_variable(
              "initial_c_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(c_value, DATA_TYPE))

        kwargs = {}
        if options is not None:
          kwargs["options"] = options
        if options_bwd is not None:
          kwargs["options_bwd"] = options_bwd

        logits = lstm_layer_function(inputs=pinputs,
                                     weights_value=weights_value,
                                     initial_state=(initial_h_state,
                                                    initial_c_state),
                                     forget_bias=forget_bias,
                                     seq_lens=pseq_lens,
                                     seq_lens_h=seq_lens,
                                     training=True,
                                     name=name,
                                     input_size=input_size,
                                     num_channels=num_channels,
                                     **kwargs)
        logits = math_ops.reduce_mean(logits, axis=0)
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      utils.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      losses = []
      inputs = _createLSTMInput(input_value, batch_size, SEQ_LEN, input_size)
      fd = {
          pinputs: inputs,
          plabels: labels_array,
      }
      if seq_lens is not None:
        fd[pseq_lens] = seq_lens

      for _ in range(0, training_steps):
        l, _ = sess.run([loss, train], fd)
        losses.append(l)
      return losses

  def _RunTrainingComparison(self,
                             name,
                             input_value,
                             forget_bias,
                             weights_value,
                             h_value,
                             c_value,
                             training_steps,
                             seq_lens=None):
    labels_array = np.ones(shape=(BATCH_SIZE,), dtype=np.int32)
    ops.reset_default_graph()
    popnn_losses = self._RunLSTMLayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        training_steps=training_steps,
        seq_lens=seq_lens,
        labels_array=labels_array,
        lstm_layer_function=self._LSTMLayer,
        device_string="/device:IPU:0")
    ops.reset_default_graph()
    ref_losses = self._RunLSTMLayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        training_steps=training_steps,
        seq_lens=seq_lens,
        labels_array=labels_array,
        lstm_layer_function=self._LSTMLayerCPU,
        device_string="/device:CPU:0")
    self.assertAllClose(popnn_losses, ref_losses)

  @parameterized.named_parameters(*LAYER_WEIGHT_CASES)
  def testLSTMLayerTraining(self, h_init, c_init):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(42)

    # Run with random weights
    for weight in np.random.rand(3):
      self._RunTrainingComparison('rand',
                                  input_value=0.,
                                  forget_bias=0.,
                                  weights_value=weight,
                                  h_value=h_init,
                                  c_value=c_init,
                                  training_steps=2)

    # seq_len=1
    for weight in np.random.rand(3):
      self._RunTrainingComparison('rand',
                                  input_value=0.,
                                  forget_bias=0.,
                                  weights_value=weight,
                                  h_value=h_init,
                                  c_value=c_init,
                                  training_steps=2,
                                  seq_lens=[1])

  @parameterized.named_parameters(*_activationTestCases())
  def testLSTMActivations(self, activation, recurrent_activation):
    input_value = 0.7
    weights_value = 0.3
    h_value = 0.5
    c_value = 10.5

    inputs = _createLSTMInput(input_value, BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    initial_state = _createLSTMInitialState(h_value, c_value, BATCH_SIZE,
                                            NUM_CHANNELS)

    forget_bias = 0.0

    def run(lstm_layer_function, act, rec_act):
      ops.reset_default_graph()
      with self.session() as sess:
        pinputs = array_ops.placeholder(DATA_TYPE,
                                        [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                        name="inputs")
        pinitial_h_state = array_ops.placeholder(DATA_TYPE,
                                                 [BATCH_SIZE, NUM_CHANNELS],
                                                 name="init_h_state")
        pinitial_c_state = array_ops.placeholder(DATA_TYPE,
                                                 [BATCH_SIZE, NUM_CHANNELS],
                                                 name="init_c_state")
        lstm_output_seq = lstm_layer_function(inputs=pinputs,
                                              weights_value=weights_value,
                                              initial_state=(pinitial_h_state,
                                                             pinitial_c_state),
                                              forget_bias=forget_bias,
                                              training=False,
                                              name=None,
                                              activation=act,
                                              recurrent_activation=rec_act)

        fd = {
            pinputs: inputs,
            pinitial_h_state: initial_state[0],
            pinitial_c_state: initial_state[1],
        }
        sess.run(variables.global_variables_initializer())
        return sess.run(lstm_output_seq, fd)

    output_cpu = run(self._LSTMLayerCPU, activation, recurrent_activation)
    output_ipu = run(self._LSTMLayer, activation, recurrent_activation)

    self.assertAllClose(output_cpu, output_ipu)

  def testLSTMCached(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      pinputs1 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs1")
      pinputs2 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs2")
      plabels = array_ops.placeholder(np.int32, [BATCH_SIZE], name="labels")

      with ops.device("/device:IPU:0"):

        def lstm_layer(inputs, name):
          initial_h_state = _get_variable(
              "initial_h_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.1, DATA_TYPE))
          initial_c_state = _get_variable(
              "initial_c_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.2, DATA_TYPE))
          return self._LSTMLayer(inputs=inputs,
                                 weights_value=1.,
                                 initial_state=(initial_h_state,
                                                initial_c_state),
                                 forget_bias=0.,
                                 training=True,
                                 name=name)

        with variable_scope.variable_scope("lstm_layer1", use_resource=True):
          logits1 = lstm_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("lstm_layer2", use_resource=True):
          logits2 = lstm_layer(pinputs2, "layer2")

        logits = (math_ops.reduce_mean(logits1, axis=0) +
                  math_ops.reduce_mean(logits2, axis=0))
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      sess.run(
          [loss, train], {
              pinputs1: _createLSTMInput(0.5, BATCH_SIZE, SEQ_LEN, INPUT_SIZE),
              pinputs2: _createLSTMInput(1.5, BATCH_SIZE, SEQ_LEN, INPUT_SIZE),
              plabels: np.ones(shape=[BATCH_SIZE], dtype=np.int32),
          })

    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_matches(report, '*/OutputGate/Op/Multiply', 1,
                                     'One fwd LSTM')
    self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 1,
                                     'One bwd LSTM')

  def testLSTMNotCached(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      # Note here the second LSTM is larger.
      pinputs1 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs1")
      pinputs2 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN * 2, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs2")
      plabels = array_ops.placeholder(np.int32, [BATCH_SIZE], name="labels")

      with ops.device("/device:IPU:0"):

        def lstm_layer(inputs, name):
          initial_h_state = _get_variable(
              "initial_h_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.1, DATA_TYPE))
          initial_c_state = _get_variable(
              "initial_c_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.2, DATA_TYPE))
          return self._LSTMLayer(inputs=inputs,
                                 weights_value=1.,
                                 initial_state=(initial_h_state,
                                                initial_c_state),
                                 forget_bias=0.,
                                 training=True,
                                 name=name)

        with variable_scope.variable_scope("lstm_layer1", use_resource=True):
          logits1 = lstm_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("lstm_layer2", use_resource=True):
          logits2 = lstm_layer(pinputs2, "layer2")

        logits = (math_ops.reduce_mean(logits1, axis=0) +
                  math_ops.reduce_mean(logits2, axis=0))
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      sess.run(
          [loss, train], {
              pinputs1: _createLSTMInput(0.5, BATCH_SIZE, SEQ_LEN, INPUT_SIZE),
              pinputs2: _createLSTMInput(1.5, BATCH_SIZE, SEQ_LEN * 2,
                                         INPUT_SIZE),
              plabels: np.ones(shape=[BATCH_SIZE], dtype=np.int32),
          })

    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_matches(report, '*/OutputGate/Op/Multiply', 2,
                                     "Two fwd LSTMs")
    self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 2,
                                     "Two bwd LSTMs")

  @parameterized.parameters((True,), (False,))
  def testLSTMWithAvailableMemoryProportionFwd(self, valid_value):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()
    h_value = 0.5
    c_value = 10.5

    with self.session() as sess:
      pinputs = array_ops.placeholder(DATA_TYPE,
                                      [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                      name="inputs")
      pinitial_h_state = array_ops.placeholder(DATA_TYPE,
                                               [BATCH_SIZE, NUM_CHANNELS],
                                               name="initial_h_state")
      pinitial_c_state = array_ops.placeholder(DATA_TYPE,
                                               [BATCH_SIZE, NUM_CHANNELS],
                                               name="initial_c_state")

      lstm_output_seq = self._LSTMLayer(
          inputs=pinputs,
          weights_value=1.,
          initial_state=(pinitial_h_state, pinitial_c_state),
          forget_bias=0.,
          training=False,
          name=None,
          options={"availableMemoryProportion": 0.7 if valid_value else -123.})

      initial_state = _createLSTMInitialState(h_value, c_value, BATCH_SIZE,
                                              NUM_CHANNELS)

      sess.run(variables.global_variables_initializer())

      def run_lstm():
        sess.run(
            lstm_output_seq, {
                pinputs: _createLSTMInput(0.7, BATCH_SIZE, SEQ_LEN,
                                          INPUT_SIZE),
                pinitial_h_state: initial_state[0],
                pinitial_c_state: initial_state[1]
            })

      if valid_value:
        run_lstm()
      else:
        self.assertRaisesRegex(errors.InternalError,
                               "Value must be greater than or equal to 0",
                               run_lstm)

  def testLSTMGreaterAvailableMemoryProportionFwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    batch_size = 256
    input_size = 256
    num_channels = 256

    h_value = 0.5
    c_value = 10.5

    def run_lstm(amp_val):
      with self.session() as sess:
        with variable_scope.variable_scope("lstm_" +
                                           str(amp_val).replace(".", "_"),
                                           use_resource=True):
          pinputs = array_ops.placeholder(DATA_TYPE,
                                          [SEQ_LEN, batch_size, input_size],
                                          name="inputs")
          pinitial_h_state = array_ops.placeholder(DATA_TYPE,
                                                   [batch_size, num_channels],
                                                   name="initial_h_state")
          pinitial_c_state = array_ops.placeholder(DATA_TYPE,
                                                   [batch_size, num_channels],
                                                   name="initial_c_state")

          lstm_output_seq = self._LSTMLayer(
              inputs=pinputs,
              weights_value=1.,
              initial_state=(pinitial_h_state, pinitial_c_state),
              forget_bias=0.,
              training=False,
              name=None,
              input_size=input_size,
              num_channels=num_channels,
              options={"availableMemoryProportion": amp_val})

        initial_state = _createLSTMInitialState(h_value, c_value, batch_size,
                                                num_channels)

        utils.move_variable_initialization_to_cpu()
        sess.run(variables.global_variables_initializer())
        sess.run(
            lstm_output_seq, {
                pinputs: _createLSTMInput(0.7, batch_size, SEQ_LEN,
                                          input_size),
                pinitial_h_state: initial_state[0],
                pinitial_c_state: initial_state[1]
            })

    run_lstm(0.8)
    run_lstm(0.1)

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_totalTileMemory(reports[0]),
                       _totalTileMemory(reports[1]))

  def _run_lstm_single_training_step(self,
                                     name,
                                     batch_size=BATCH_SIZE,
                                     input_size=INPUT_SIZE,
                                     num_channels=NUM_CHANNELS,
                                     amp_val=None):
    self._RunLSTMLayerTraining(
        name=name,
        input_value=0.,
        forget_bias=0.,
        weights_value=0.7,
        h_value=0.5,
        c_value=10.5,
        training_steps=1,
        seq_lens=None,
        labels_array=np.ones(shape=(batch_size,), dtype=np.int32),
        lstm_layer_function=self._LSTMLayer,
        device_string="/device:IPU:0",
        batch_size=batch_size,
        input_size=input_size,
        num_channels=num_channels,
        options_bwd={"availableMemoryProportion": amp_val})

  @parameterized.parameters((True,), (False,))
  def testLSTMWithAvailableMemoryProportionBwd(self, valid_value):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    name = ("" if valid_value else "in") + "validAvailableMemoryProportionBwd"

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      if valid_value:
        self._run_lstm_single_training_step(name=name, amp_val=0.7)
      else:
        with self.assertRaisesRegex(
            errors.InternalError, "Value must be greater than or equal to 0"):
          self._run_lstm_single_training_step(name=name, amp_val=-123.)

  def testLSTMGreaterAvailableMemoryProportionBwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportionBwd"
    batch_size = 256
    input_size = 256
    num_channels = 256

    self._run_lstm_single_training_step(name=name,
                                        batch_size=batch_size,
                                        input_size=input_size,
                                        num_channels=num_channels,
                                        amp_val=0.8)
    self._run_lstm_single_training_step(name=name,
                                        batch_size=batch_size,
                                        input_size=input_size,
                                        num_channels=num_channels,
                                        amp_val=0.1)

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_totalTileMemory(reports[0]),
                       _totalTileMemory(reports[1]))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

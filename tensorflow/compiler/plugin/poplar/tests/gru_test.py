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
import json
import numpy as np
from absl.testing import parameterized
import pva

# pylint: disable=unused-import
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.keras.layers import GRU
# pylint: enable=unused-import

DATA_TYPE = np.float32
BATCH_SIZE = 1
SEQ_LEN = 3
INPUT_SIZE = 5
NUM_CHANNELS = 8


def _totalTileMemory(report):
  return sum(tile.memory.total.excludingGaps
             for tile in report.compilation.tiles)


class AUGRUCell(rnn_cell.RNNCell):
  def __init__(self, num_units, kernel_init, recurrent_init, bias_init):
    super().__init__()
    self._num_units = num_units
    self.kernel_init = kernel_init
    self.recurrent_init = recurrent_init
    self.bias_init = bias_init

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    # Unpack inputs and attention scores
    x = inputs[:, :-1]
    a = inputs[:, -1]

    # Get weights
    n = self._num_units
    h = state
    input_dim = x.shape[1]
    with variable_scope.variable_scope("",
                                       use_resource=True,
                                       reuse=variable_scope.AUTO_REUSE):
      kernel = _get_variable('kernel', [input_dim, n * 3], self.kernel_init)
      rec_kernel = _get_variable('recurrent_kernel', [n, n * 3],
                                 self.recurrent_init)
      bias = _get_variable('bias', [n * 3], self.bias_init)

    # Reset gate
    rx = math_ops.matmul(x, kernel[:, :n])
    rh = math_ops.matmul(h, rec_kernel[:, :n])
    r = math_ops.sigmoid(rx + rh + bias[:n])

    # Update gate
    zx = math_ops.matmul(x, kernel[:, n:n * 2])
    zh = math_ops.matmul(h, rec_kernel[:, n:n * 2])
    z = math_ops.sigmoid(zx + zh + bias[n:n * 2])

    # Candidate state
    cx = math_ops.matmul(x, kernel[:, n * 2:])
    ch = math_ops.matmul(r * h, rec_kernel[:, n * 2:])
    c = math_ops.tanh(cx + ch + bias[n * 2:])

    # Attention score influences the mixing of the old + candidate states.
    # This is the only difference between the GRU and AUGRU cells.
    z = z * (1 - a)

    # Mix old state and candidate state
    h = z * h + (1 - z) * c
    return h, h


def _get_variable(name, shape, initializer):
  return variable_scope.get_variable(name,
                                     shape=shape,
                                     initializer=initializer,
                                     dtype=DATA_TYPE)


def _createGRUInput(value, shape):
  return np.full(fill_value=value, shape=shape, dtype=DATA_TYPE)


def _createGRUInitialState(value, shape):
  return np.full(fill_value=value, shape=shape, dtype=DATA_TYPE)


class GRUTest(xla_test.XLATestCase, parameterized.TestCase):  # pylint: disable=abstract-method
  def _GRULayerCPU(
      self,
      inputs,
      weights_value,
      seq_length,
      seq_val,
      initial_state,
      att_scores,
      training,
      name,
      input_size=INPUT_SIZE,  # pylint: disable=unused-argument
      num_channels=NUM_CHANNELS,
      activation='tanh',
      recurrent_activation='sigmoid'):
    del name
    with ops.device("/device:CPU:0"):
      kernel_init = init_ops.constant_initializer(weights_value, DATA_TYPE)
      recurrent_init = init_ops.constant_initializer(weights_value, DATA_TYPE)
      bias_init = init_ops.constant_initializer(0.0, DATA_TYPE)
      if att_scores is None:
        gru = GRU(num_channels,
                  activation=activation,
                  recurrent_activation=recurrent_activation,
                  kernel_initializer=kernel_init,
                  recurrent_initializer=recurrent_init,
                  bias_initializer=bias_init,
                  time_major=True,
                  return_sequences=True,
                  stateful=True,
                  reset_after=False)
        outputs = gru(inputs, initial_state=initial_state, training=training)
      else:
        # There is no native AUGRU implementation
        inputs = array_ops.concat(
            [inputs, array_ops.expand_dims(att_scores, -1)], axis=2)
        outputs, _ = rnn.dynamic_rnn(AUGRUCell(num_channels, kernel_init,
                                               recurrent_init, bias_init),
                                     inputs=inputs,
                                     sequence_length=seq_length,
                                     initial_state=initial_state,
                                     dtype=DATA_TYPE,
                                     scope="augru",
                                     time_major=True)

      outputs = outputs if seq_val is None else outputs[0:min(
          SEQ_LEN, seq_val[0])]
      return outputs

  def _GRULayer(self,
                inputs,
                weights_value,
                seq_length,
                seq_val,
                initial_state,
                att_scores,
                training,
                name,
                activation='tanh',
                recurrent_activation='sigmoid',
                input_size=INPUT_SIZE,
                num_channels=NUM_CHANNELS,
                options=None,
                options_bwd=None):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("gru_layer", use_resource=True):
        kernel = _get_variable(
            "kernel",
            shape=[input_size + num_channels, 3 * num_channels],
            initializer=init_ops.constant_initializer(weights_value,
                                                      DATA_TYPE))
        biases = _get_variable("biases",
                               shape=[3, num_channels],
                               initializer=init_ops.constant_initializer(
                                   0.0, DATA_TYPE))
      options = {} if options is None else options
      options_bwd = {} if options is None else options_bwd
      if seq_length is None:
        outputs, _, _ = gen_popnn_ops.popnn_gru_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            initial_state=initial_state,
            is_training=training,
            name=name,
            options=json.dumps(options),
            options_bwd=json.dumps(options_bwd))
      elif att_scores is not None:
        outputs, _, _ = gen_popnn_ops.popnn_augru_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            initial_state=initial_state,
            is_training=training,
            seq_len=seq_length,
            att_score=att_scores,
            name=name,
            options=json.dumps(options),
            options_bwd=json.dumps(options_bwd))
      else:
        outputs, _, _ = gen_popnn_ops.popnn_dynamic_gru_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            initial_state=initial_state,
            is_training=training,
            seq_len=seq_length,
            name=name,
            options=json.dumps(options),
            options_bwd=json.dumps(options_bwd))
      outputs = outputs if seq_val is None else outputs[0:min(
          SEQ_LEN, seq_val[0])]
      return outputs

  def _RunGRULayerInference(self, name, input_value, weights_value, seq_val,
                            init_state_value, att_score_val,
                            gru_layer_function):
    with self.session() as sess:
      pinputs = array_ops.placeholder(DATA_TYPE,
                                      [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                      name="inputs")
      pinitial_state = array_ops.placeholder(DATA_TYPE,
                                             [BATCH_SIZE, NUM_CHANNELS],
                                             name="initial_state")
      pseq_len = array_ops.placeholder(
          np.int32, [BATCH_SIZE],
          name="seq_len") if seq_val is not None else None

      patt_scores = array_ops.placeholder(
          DATA_TYPE, [SEQ_LEN, BATCH_SIZE],
          name="att_score") if att_score_val is not None else None

      gru_output_seq = gru_layer_function(inputs=pinputs,
                                          weights_value=weights_value,
                                          seq_length=pseq_len,
                                          att_scores=patt_scores,
                                          seq_val=seq_val,
                                          initial_state=pinitial_state,
                                          training=False,
                                          name=name)

      inputs = _createGRUInput(input_value, pinputs.shape)
      initial_state = _createGRUInitialState(init_state_value,
                                             pinitial_state.shape)
      fd = {pinputs: inputs, pinitial_state: initial_state}
      if pseq_len is not None:
        fd[pseq_len] = seq_val
      if patt_scores is not None:
        fd[patt_scores] = np.full(patt_scores.shape, att_score_val, DATA_TYPE)

      sess.run(variables.global_variables_initializer())
      return sess.run(gru_output_seq, fd)

  def _RunInferenceComparison(self,
                              name,
                              input_value,
                              weights_value,
                              init_state_value,
                              seq_val=None,
                              att_score_val=None):
    ops.reset_default_graph()
    popnn_out = self._RunGRULayerInference(name=name,
                                           input_value=input_value,
                                           weights_value=weights_value,
                                           seq_val=seq_val,
                                           att_score_val=att_score_val,
                                           init_state_value=init_state_value,
                                           gru_layer_function=self._GRULayer)
    ref_out = self._RunGRULayerInference(name=name,
                                         input_value=input_value,
                                         weights_value=weights_value,
                                         seq_val=seq_val,
                                         att_score_val=att_score_val,
                                         init_state_value=init_state_value,
                                         gru_layer_function=self._GRULayerCPU)
    # Check that the whole output sequence matches
    self.assertAllClose(popnn_out, ref_out)

  def testGRULayerInference(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(0)
    # Run with attention scores (augru):
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('augru',
                                   input_value=0.01,
                                   weights_value=0.1,
                                   init_state_value=init_state_value,
                                   seq_val=[1],
                                   att_score_val=0.5)

    # Run with all-0 weights
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=0.,
                                   init_state_value=init_state_value)

    # Run with all-1 weights
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=1.,
                                   init_state_value=init_state_value)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunInferenceComparison('rand',
                                     input_value=0.,
                                     weights_value=weight,
                                     init_state_value=init_state_value)

    # Run with '1'' seq_len
    assert BATCH_SIZE == 1
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=0.,
                                   init_state_value=init_state_value,
                                   seq_val=[1])

    # Run with zero seq_len
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=0.,
                                   init_state_value=init_state_value,
                                   seq_val=[0])

  def _RunGRULayerTraining(self,
                           name,
                           input_value,
                           weights_value,
                           seq_val,
                           init_state_value,
                           training_steps,
                           labels_array,
                           att_score_val,
                           gru_layer_function,
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

      pseq_len = array_ops.placeholder(
          np.int32, [batch_size],
          name="seq_len") if seq_val is not None else None

      patt_scores = array_ops.placeholder(
          DATA_TYPE, [SEQ_LEN, batch_size],
          name="att_score") if att_score_val is not None else None

      with ops.device(device_string):
        with variable_scope.variable_scope("gru_layer", use_resource=True):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(
                  init_state_value, DATA_TYPE))

        kwargs = {}
        if options is not None:
          kwargs["options"] = options
        if options_bwd is not None:
          kwargs["options_bwd"] = options_bwd

        logits = gru_layer_function(inputs=pinputs,
                                    weights_value=weights_value,
                                    seq_length=pseq_len,
                                    seq_val=seq_val,
                                    initial_state=initial_state,
                                    att_scores=patt_scores,
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
      inputs = _createGRUInput(input_value, pinputs.shape)
      fd = {
          pinputs: inputs,
          plabels: labels_array,
      }
      if seq_val is not None:
        fd[pseq_len] = seq_val
      if patt_scores is not None:
        fd[patt_scores] = np.full(patt_scores.shape, att_score_val, DATA_TYPE)

      for _ in range(0, training_steps):
        l, _ = sess.run([loss, train], fd)
        losses.append(l)
      return losses

  def _RunTrainingComparison(self,
                             name,
                             input_value,
                             weights_value,
                             init_state_value,
                             training_steps,
                             seq_val=None,
                             att_score_val=None):
    labels_array = np.ones(shape=[BATCH_SIZE], dtype=np.int32)
    ops.reset_default_graph()
    popnn_losses = self._RunGRULayerTraining(name=name,
                                             input_value=input_value,
                                             weights_value=weights_value,
                                             seq_val=seq_val,
                                             init_state_value=init_state_value,
                                             att_score_val=att_score_val,
                                             training_steps=training_steps,
                                             labels_array=labels_array,
                                             gru_layer_function=self._GRULayer,
                                             device_string="/device:IPU:0")
    ops.reset_default_graph()
    ref_losses = self._RunGRULayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        seq_val=seq_val,
        init_state_value=init_state_value,
        att_score_val=att_score_val,
        training_steps=training_steps,
        labels_array=labels_array,
        gru_layer_function=self._GRULayerCPU,
        device_string="/device:CPU:0")
    self.assertAllClose(popnn_losses, ref_losses)

  def testGRULayerTraining(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    np.random.seed(42)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('rand',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3)

    # Run with a sequence length
    assert BATCH_SIZE == 1
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('rand',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3,
                                    seq_val=[1])

    # Run with attention scores
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('augru',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3,
                                    seq_val=[1],
                                    att_score_val=0.5)

  def testGRUActivations(self):
    input_value = 0.7
    weights_value = 0.3
    init_state_value = 1.
    seq_val = None

    inputs = _createGRUInput(input_value, [SEQ_LEN, BATCH_SIZE, INPUT_SIZE])
    initial_state = _createGRUInitialState(init_state_value,
                                           [BATCH_SIZE, NUM_CHANNELS])

    def run(gru_layer_function, act, rec_act):
      ops.reset_default_graph()
      with self.session() as sess:
        pinputs = array_ops.placeholder(DATA_TYPE,
                                        [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                        name="inputs")
        pinitial_state = array_ops.placeholder(DATA_TYPE,
                                               [BATCH_SIZE, NUM_CHANNELS],
                                               name="initial_state")
        pseq_len = array_ops.placeholder(
            np.int32, [BATCH_SIZE],
            name="seq_len") if seq_val is not None else None

        gru_output_seq = gru_layer_function(inputs=pinputs,
                                            weights_value=weights_value,
                                            seq_length=pseq_len,
                                            seq_val=seq_val,
                                            att_scores=None,
                                            initial_state=pinitial_state,
                                            training=False,
                                            name=None,
                                            activation=act,
                                            recurrent_activation=rec_act)

        fd = {pinputs: inputs, pinitial_state: initial_state}
        if pseq_len is not None:
          fd[pseq_len] = seq_val
        sess.run(variables.global_variables_initializer())
        return sess.run(gru_output_seq, fd)

    for activation in ['tanh', 'relu', 'softmax', 'sigmoid', 'hard_sigmoid']:
      for recurrent_activation in ['softmax', 'sigmoid', 'hard_sigmoid']:
        output_cpu = run(self._GRULayerCPU, activation, recurrent_activation)
        output_ipu = run(self._GRULayer, activation, recurrent_activation)

        self.assertAllClose(output_cpu, output_ipu)

  def testGRUCached(self):
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

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.1, DATA_TYPE))
          return self._GRULayer(inputs=inputs,
                                weights_value=1.,
                                seq_length=None,
                                seq_val=None,
                                att_scores=None,
                                initial_state=initial_state,
                                training=True,
                                name=name)

        with variable_scope.variable_scope("gru_layer1", use_resource=True):
          logits1 = gru_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("gru_layer2", use_resource=True):
          logits2 = gru_layer(pinputs2, "layer2")

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
              pinputs1: _createGRUInput(0.5, pinputs1.shape),
              pinputs2: _createGRUInput(1.5, pinputs2.shape),
              plabels: np.ones(shape=[BATCH_SIZE], dtype=np.int32),
          })

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*BasicGruCell/ProcessUnits/Weight/Conv*/Convolve', 2,
          "There should be two fwd GRUs")
      self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 1,
                                       "There should be one bwd GRU")

  def testGRUNotCached(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      # Note here the second GRU is larger.
      pinputs1 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs1")
      pinputs2 = array_ops.placeholder(DATA_TYPE,
                                       [SEQ_LEN * 2, BATCH_SIZE, INPUT_SIZE],
                                       name="inputs2")
      plabels = array_ops.placeholder(np.int32, [BATCH_SIZE], name="labels")

      with ops.device("/device:IPU:0"):

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[BATCH_SIZE, NUM_CHANNELS],
              initializer=init_ops.constant_initializer(0.1, DATA_TYPE))
          return self._GRULayer(inputs=inputs,
                                weights_value=1.,
                                seq_length=None,
                                seq_val=None,
                                att_scores=None,
                                initial_state=initial_state,
                                training=True,
                                name=name)

        with variable_scope.variable_scope("gru_layer1", use_resource=True):
          logits1 = gru_layer(pinputs1, "layer1")
        with variable_scope.variable_scope("gru_layer2", use_resource=True):
          logits2 = gru_layer(pinputs2, "layer2")

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
              pinputs1: _createGRUInput(0.5, pinputs1.shape),
              pinputs2: _createGRUInput(1.5, pinputs2.shape),
              plabels: np.ones(shape=[BATCH_SIZE], dtype=np.int32),
          })

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*BasicGruCell/ProcessUnits/Weight/Conv*/Convolve', 4,
          "There should be four fwd GRUs")
      self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 2,
                                       "There should be two bwd GRUs")

  @parameterized.parameters((True,), (False,))
  def testGRUWithAvailableMemoryProportionFwd(self, valid_value):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      pinputs = array_ops.placeholder(DATA_TYPE,
                                      [SEQ_LEN, BATCH_SIZE, INPUT_SIZE],
                                      name="inputs")
      pinitial_state = array_ops.placeholder(DATA_TYPE,
                                             [BATCH_SIZE, NUM_CHANNELS],
                                             name="initial_state")

      gru_output_seq = self._GRULayer(
          inputs=pinputs,
          weights_value=1.,
          seq_length=None,
          seq_val=None,
          att_scores=None,
          initial_state=pinitial_state,
          training=False,
          name=None,
          options={"availableMemoryProportion": 0.7 if valid_value else -123.})

      sess.run(variables.global_variables_initializer())

      def run_gru():
        sess.run(
            gru_output_seq, {
                pinputs: _createGRUInput(0.7, pinputs.shape),
                pinitial_state: _createGRUInitialState(1.,
                                                       pinitial_state.shape)
            })

      if valid_value:
        run_gru()
      else:
        self.assertRaisesRegex(errors.InternalError,
                               "Value must be greater than or equal to 0",
                               run_gru)

  def testGRUGreaterAvailableMemoryProportionFwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportion"
    batch_size = 256
    input_size = 256
    num_channels = 256

    def run_gru(amp_val):
      with self.session() as sess:
        with variable_scope.variable_scope("gru_" +
                                           str(amp_val).replace(".", "_"),
                                           use_resource=True):
          pinputs = array_ops.placeholder(DATA_TYPE,
                                          [SEQ_LEN, batch_size, input_size],
                                          name="inputs")
          pinitial_state = array_ops.placeholder(DATA_TYPE,
                                                 [batch_size, num_channels],
                                                 name="initial_state")

        gru_output_seq = self._GRULayer(
            inputs=pinputs,
            weights_value=1.,
            seq_length=None,
            seq_val=None,
            att_scores=None,
            initial_state=pinitial_state,
            training=False,
            name=name,
            input_size=input_size,
            num_channels=num_channels,
            options={"availableMemoryProportion": amp_val})

        utils.move_variable_initialization_to_cpu()
        sess.run(variables.global_variables_initializer())
        sess.run(
            gru_output_seq, {
                pinputs: _createGRUInput(0.7, pinputs.shape),
                pinitial_state: _createGRUInitialState(1.,
                                                       pinitial_state.shape)
            })

    run_gru(0.8)
    run_gru(0.1)

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_totalTileMemory(reports[0]),
                       _totalTileMemory(reports[1]))

  def _run_single_gru_training_step(self,
                                    name,
                                    batch_size=BATCH_SIZE,
                                    input_size=INPUT_SIZE,
                                    num_channels=NUM_CHANNELS,
                                    amp_val=None):
    self._RunGRULayerTraining(
        name=name,
        input_value=0.,
        weights_value=0.7,
        init_state_value=1.,
        training_steps=1,
        seq_val=None,
        att_score_val=0.5,
        labels_array=np.ones(shape=[batch_size], dtype=np.int32),
        gru_layer_function=self._GRULayer,
        device_string="/device:IPU:0",
        batch_size=batch_size,
        input_size=input_size,
        num_channels=num_channels,
        options_bwd={"availableMemoryProportion": amp_val})

  @parameterized.parameters((True), (False))
  def testGRUWithAvailableMemoryProportionBwd(self, valid_value):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    name = ("" if valid_value else "in") + "validAvailableMemoryProportionBwd"

    if valid_value:
      self._run_single_gru_training_step(name, amp_val=0.7)
    else:
      with self.assertRaisesRegex(errors.InternalError,
                                  "Value must be greater than or equal to 0"):
        self._run_single_gru_training_step(name, amp_val=-123.)

  def testGRUGreaterAvailableMemoryProportionBwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportion"
    batch_size = 256
    input_size = 256
    num_channels = 256

    self._run_single_gru_training_step(name,
                                       batch_size=batch_size,
                                       input_size=input_size,
                                       num_channels=num_channels,
                                       amp_val=0.8)
    self._run_single_gru_training_step(name,
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

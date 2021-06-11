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
import pva
from test_utils import ReportJSON, ReportHelper

# pylint: disable=unused-import
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.keras.layers import GRU
# pylint: enable=unused-import

dataType = np.float32
batch_size = 1
seq_len = 3
input_size = 5
num_channels = 8


def _get_variable(name, shape, initializer):
  return variable_scope.get_variable(name,
                                     shape=shape,
                                     initializer=initializer,
                                     dtype=dataType)


def _createGRUInput(value, shape):
  return np.full(fill_value=value, shape=shape, dtype=dataType)


def _createGRUInitialState(value, shape):
  return np.full(fill_value=value, shape=shape, dtype=dataType)


class GRUTest(xla_test.XLATestCase):
  def _GRULayerCPU(self,
                   inputs,
                   weights_value,
                   seq_length,
                   seq_val,
                   initial_state,
                   training,
                   name,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   output_full_sequence=True):
    #pylint: disable=unused-argument
    del name
    with ops.device("/device:CPU:0"):
      gru = GRU(num_channels,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_initializer=init_ops.constant_initializer(
                    weights_value, dataType),
                recurrent_initializer=init_ops.constant_initializer(
                    weights_value, dataType),
                bias_initializer=init_ops.constant_initializer(0.0, dataType),
                time_major=True,
                return_sequences=output_full_sequence,
                stateful=True,
                reset_after=False)
      outputs = gru(inputs, initial_state=initial_state, training=training)
      outputs = outputs if seq_val is None else outputs[0:min(
          seq_len, seq_val[0])]
      return outputs

  def _GRULayer(self,
                inputs,
                weights_value,
                seq_length,
                seq_val,
                initial_state,
                training,
                name,
                activation='tanh',
                recurrent_activation='sigmoid',
                output_full_sequence=True):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("gru_layer", use_resource=True):
        kernel = _get_variable(
            "kernel",
            shape=[input_size + num_channels, 3 * num_channels],
            initializer=init_ops.constant_initializer(weights_value, dataType))
        biases = _get_variable("biases",
                               shape=[3, num_channels],
                               initializer=init_ops.constant_initializer(
                                   0.0, dataType))
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
            output_full_sequence=output_full_sequence,
            name=name)
      else:
        outputs, _, _ = gen_popnn_ops.popnn_dynamic_gru_layer(
            activation=activation,
            recurrent_activation=recurrent_activation,
            inputs=inputs,
            seq_len=seq_length,
            num_channels=num_channels,
            kernel=kernel,
            biases=biases,
            initial_state=initial_state,
            is_training=training,
            name=name)
      outputs = outputs if seq_val is None else outputs[0:min(
          seq_len, seq_val[0])]
      return outputs

  def _RunGRULayerInference(self, name, input_value, weights_value, seq_val,
                            init_state_value, output_full_sequence,
                            gru_layer_function):
    with self.session() as sess:
      pinputs = array_ops.placeholder(dataType,
                                      [seq_len, batch_size, input_size],
                                      name="inputs")
      pinitial_state = array_ops.placeholder(dataType,
                                             [batch_size, num_channels],
                                             name="initial_state")
      pseq_len = array_ops.placeholder(
          np.int32, [batch_size],
          name="seq_len") if seq_val is not None else None

      gru_output_seq = gru_layer_function(
          inputs=pinputs,
          weights_value=weights_value,
          seq_length=pseq_len,
          seq_val=seq_val,
          initial_state=pinitial_state,
          training=False,
          output_full_sequence=output_full_sequence,
          name=name)

      inputs = _createGRUInput(input_value, pinputs.shape)
      initial_state = _createGRUInitialState(init_state_value,
                                             pinitial_state.shape)
      fd = {
          pinputs: inputs,
          pinitial_state: initial_state,
      }
      if pseq_len is not None:
        fd[pseq_len] = seq_val

      sess.run(variables.global_variables_initializer())
      return sess.run(gru_output_seq, fd)

  def _RunInferenceComparison(self,
                              name,
                              input_value,
                              weights_value,
                              init_state_value,
                              seq_val=None,
                              output_full_sequence=True):
    ops.reset_default_graph()
    popnn_out = self._RunGRULayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        seq_val=seq_val,
        init_state_value=init_state_value,
        output_full_sequence=output_full_sequence,
        gru_layer_function=self._GRULayer)
    ref_out = self._RunGRULayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        seq_val=seq_val,
        init_state_value=init_state_value,
        output_full_sequence=output_full_sequence,
        gru_layer_function=self._GRULayerCPU)
    # Check that the whole output sequence matches
    self.assertAllClose(popnn_out, ref_out)

  def testGRULayerInference(self):
    ReportJSON(self)
    np.random.seed(0)

    # Run with outputFullSequence false
    weight0 = 1.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('not-full-sequence',
                                   input_value=0.,
                                   weights_value=weight0,
                                   init_state_value=init_state_value,
                                   output_full_sequence=False)

    # Run with all-0 weights
    weight0 = 0.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=weight0,
                                   init_state_value=init_state_value)

    # Run with all-1 weights
    weight1 = 1.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=weight1,
                                   init_state_value=init_state_value)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunInferenceComparison('rand',
                                     input_value=0.,
                                     weights_value=weight,
                                     init_state_value=init_state_value)

    # Run with '1' seq_len
    assert batch_size == 1
    weight0 = 0.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=weight0,
                                   init_state_value=init_state_value,
                                   seq_val=[1])

    # Run with zero seq_len
    weight0 = 0.
    for init_state_value in [0., 1.]:
      self._RunInferenceComparison('ones',
                                   input_value=0.,
                                   weights_value=weight0,
                                   init_state_value=init_state_value,
                                   seq_val=[0])

  def _RunGRULayerTraining(self, name, input_value, weights_value, seq_val,
                           init_state_value, training_steps, labels_array,
                           output_full_sequence, gru_layer_function,
                           device_string):
    with self.session() as sess:
      pinputs = array_ops.placeholder(dataType,
                                      [seq_len, batch_size, input_size],
                                      name="inputs")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      pseq_len = array_ops.placeholder(
          np.int32, [batch_size],
          name="seq_len") if seq_val is not None else None

      with ops.device(device_string):
        with variable_scope.variable_scope("gru_layer", use_resource=True):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(
                  init_state_value, dataType))
        logits = gru_layer_function(inputs=pinputs,
                                    weights_value=weights_value,
                                    seq_length=pseq_len,
                                    seq_val=seq_val,
                                    initial_state=initial_state,
                                    training=True,
                                    output_full_sequence=output_full_sequence,
                                    name=name)
        # Average over sequence
        if output_full_sequence:
          logits = math_ops.reduce_mean(logits, axis=0)
        softmax = nn.sparse_softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=array_ops.stop_gradient(plabels))
        loss = math_ops.reduce_mean(softmax)
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

      sess.run(variables.global_variables_initializer())
      losses = []
      inputs = _createGRUInput(input_value, pinputs.shape)
      fd = {
          pinputs: inputs,
          plabels: labels_array,
      }
      if seq_val is not None:
        fd[pseq_len] = seq_val

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
                             output_full_sequence=True):
    labels_array = np.ones(shape=[batch_size], dtype=np.int32)
    ops.reset_default_graph()
    popnn_losses = self._RunGRULayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        seq_val=seq_val,
        init_state_value=init_state_value,
        training_steps=training_steps,
        labels_array=labels_array,
        output_full_sequence=output_full_sequence,
        gru_layer_function=self._GRULayer,
        device_string="/device:IPU:0")
    ops.reset_default_graph()
    ref_losses = self._RunGRULayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        seq_val=seq_val,
        init_state_value=init_state_value,
        training_steps=training_steps,
        labels_array=labels_array,
        output_full_sequence=output_full_sequence,
        gru_layer_function=self._GRULayerCPU,
        device_string="/device:CPU:0")
    self.assertAllClose(popnn_losses, ref_losses)

  def testGRULayerTraining(self):
    ReportJSON(self)
    np.random.seed(42)

    # Run with random weights
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('rand',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3)

    # Run with outputFullSequence false
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('rand',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3,
                                    output_full_sequence=False)

    # Run with a sequence length
    assert batch_size == 1
    for weight in np.random.rand(3):
      for init_state_value in [0., 1.]:
        self._RunTrainingComparison('rand',
                                    input_value=0.,
                                    weights_value=weight,
                                    init_state_value=init_state_value,
                                    training_steps=3,
                                    seq_val=[1])

  def testGRUActivations(self):
    input_value = 0.7
    weights_value = 0.3
    init_state_value = 1.
    seq_val = None

    inputs = _createGRUInput(input_value, [seq_len, batch_size, input_size])
    initial_state = _createGRUInitialState(init_state_value,
                                           [batch_size, num_channels])

    def run(gru_layer_function, act, rec_act):
      ops.reset_default_graph()
      with self.session() as sess:
        pinputs = array_ops.placeholder(dataType,
                                        [seq_len, batch_size, input_size],
                                        name="inputs")
        pinitial_state = array_ops.placeholder(dataType,
                                               [batch_size, num_channels],
                                               name="initial_state")
        pseq_len = array_ops.placeholder(
            np.int32, [batch_size],
            name="seq_len") if seq_val is not None else None

        gru_output_seq = gru_layer_function(inputs=pinputs,
                                            weights_value=weights_value,
                                            seq_length=pseq_len,
                                            seq_val=seq_val,
                                            initial_state=pinitial_state,
                                            training=False,
                                            name=None,
                                            activation=act,
                                            output_full_sequence=True,
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
    report_helper = ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      pinputs1 = array_ops.placeholder(dataType,
                                       [seq_len, batch_size, input_size],
                                       name="inputs1")
      pinputs2 = array_ops.placeholder(dataType,
                                       [seq_len, batch_size, input_size],
                                       name="inputs2")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      with ops.device("/device:IPU:0"):

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(0.1, dataType))
          return self._GRULayer(inputs=inputs,
                                weights_value=1.,
                                seq_length=None,
                                seq_val=None,
                                initial_state=initial_state,
                                training=True,
                                output_full_sequence=True,
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
              plabels: np.ones(shape=[batch_size], dtype=np.int32),
          })

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*BasicGruCell/ProcessUnits/Weight/Conv*/Convolve', 2,
          "There should be two fwd GRUs")
      self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 1,
                                       "There should be one bwd GRU")

  def testGRUNotCached(self):
    cfg = IPUConfig()
    report_helper = ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      # Note here the second GRU is larger.
      pinputs1 = array_ops.placeholder(dataType,
                                       [seq_len, batch_size, input_size],
                                       name="inputs1")
      pinputs2 = array_ops.placeholder(dataType,
                                       [seq_len * 2, batch_size, input_size],
                                       name="inputs2")
      plabels = array_ops.placeholder(np.int32, [batch_size], name="labels")

      with ops.device("/device:IPU:0"):

        def gru_layer(inputs, name):
          initial_state = _get_variable(
              "initial_state",
              shape=[batch_size, num_channels],
              initializer=init_ops.constant_initializer(0.1, dataType))
          return self._GRULayer(inputs=inputs,
                                weights_value=1.,
                                seq_length=None,
                                seq_val=None,
                                initial_state=initial_state,
                                training=True,
                                output_full_sequence=True,
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
              plabels: np.ones(shape=[batch_size], dtype=np.int32),
          })

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*BasicGruCell/ProcessUnits/Weight/Conv*/Convolve', 4,
          "There should be four fwd GRUs")
      self.assert_compute_sets_matches(report, '*/MulOGate/Op/Multiply', 2,
                                       "There should be two bwd GRUs")


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

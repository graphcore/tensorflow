# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Test for IPU CTC Loss function."""

import numpy as np
from absl.testing import parameterized
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data import Dataset
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ctc_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow import sparse
from tensorflow.nn import ctc_beam_search_decoder as cpu_ctc_beam_search_decoder

dataType = np.float32


# Only works using logits
class CTCPredictionsCpu(layers.Layer):
  def __init__(self, blank_index=0, beam_width=100, top_paths=1, name=None):
    super().__init__(name)
    self.blank_index = blank_index
    self.beam_width = beam_width
    self.top_paths = top_paths

  @staticmethod
  def mask_out_junk_values(predictions, blank_index, max_time):
    predictions = [
        sparse.to_dense(t, default_value=blank_index) for t in predictions
    ]
    for i, t in enumerate(predictions):
      paddings = [[0, 0], [0, max_time - array_ops.shape(t)[1]]]
      t = array_ops.pad(t, paddings, constant_values=blank_index)
      predictions[i] = array_ops.reshape(t,
                                         [array_ops.shape(t)[0], 1, max_time])
    return array_ops.concat(predictions, 1)

  def call(self, data, data_length, **kwargs):  # pylint: disable=W0221

    predictions, probs = cpu_ctc_beam_search_decoder(
        data,
        data_length,
        beam_width=self.beam_width,
        top_paths=self.top_paths)
    predictions = self.mask_out_junk_values(predictions, self.blank_index,
                                            array_ops.shape(data)[0])

    predictions = ipu.keras.layers.CTCPredictionsLayer._select_most_likely_path(  # pylint: disable=protected-access
        probs, predictions, self.top_paths, None)
    return predictions


class CTCLossEndpointCpu(layers.Layer):
  def __init__(self, blank_index=0, name=None):
    super().__init__(name=name)
    self.blank_index = blank_index

  def call(self,
           logits,
           logit_length,
           labels=None,
           label_length=None,
           **kwargs):  # pylint: disable=W0221
    loss = ctc_ops.ctc_loss_v2(labels,
                               logits,
                               label_length,
                               logit_length,
                               blank_index=self.blank_index)
    loss = math_ops.reduce_mean(loss)
    self.add_loss(loss)
    return logits


class CTCBeamSearchParams:
  def __init__(self, params):
    self.params = params


def generate_beam_search_params():
  A = CTCBeamSearchParams({
      "batch_size": 4,
      "actual_label_length": 2,
      # randomly generated labels need 2n+1 time steps because there are
      # implicit blank steps around repeated labels
      "max_label_length": 5,
      "max_time": 5,
      "num_classes": 4,
      "blank_index": 3,  # blank index should be num_classes - 1
      "top_paths": 1,  # top paths must be less than beam width
      "beam_width": 2,
      "seed": 9,
  })
  B = CTCBeamSearchParams(dict(A.params))
  B.params["top_paths"] = 3
  B.params["beam_width"] = 4
  B.params["seed"] = 10
  return [("Greedy", A), ("TopPaths", B)]


class CTCLossTest(test.TestCase, parameterized.TestCase):
  @staticmethod
  def get_params():
    return {
        "batch_size": 4,
        "actual_label_length": 2,
        # randomly generated labels need 2n+1 time steps because there are
        # implicit blank steps around repeated labels
        "max_label_length": 5,
        "max_time": 5,
        "num_classes": 4,
        "blank_index": 0,
    }

  @staticmethod
  def create_predictions_model(predictions_layer,
                               ctc_params,
                               log_softmax=False):
    batch_size = ctc_params["batch_size"]
    num_classes = ctc_params["num_classes"]
    max_time = ctc_params["max_time"]

    logits = keras.layers.Input((max_time, num_classes),
                                batch_size=batch_size,
                                dtype=np.float32,
                                name="logits")
    logit_length = keras.layers.Input((),
                                      batch_size=batch_size,
                                      dtype=np.int32,
                                      name="logit_length")
    x = logits

    dense_layer = layers.Dense(num_classes,
                               activation='relu',
                               kernel_initializer="identity",
                               bias_initializer='ones')
    x = dense_layer(x)

    transpose_fn = lambda x: keras.backend.permute_dimensions(x, (1, 0, 2))
    transpose_layer = layers.Lambda(transpose_fn)
    x = transpose_layer(x)

    if log_softmax:
      log_softmax_fn = lambda x: nn_ops.log_softmax_v2(x, axis=2)
      log_softmax_layer = layers.Lambda(log_softmax_fn)
      x = log_softmax_layer(x)

    predictions = predictions_layer(x, logit_length)
    m = keras.Model(inputs=[logits, logit_length], outputs=predictions)
    # No need to compile as we don't need any metrics, only the output.
    return m

  @staticmethod
  def create_loss_model(loss_layer, ctc_params, log_softmax=False):
    batch_size = ctc_params["batch_size"]
    num_classes = ctc_params["num_classes"]
    max_label_length = ctc_params["max_label_length"]
    max_time = ctc_params["max_time"]

    labels = keras.layers.Input((max_label_length),
                                batch_size=batch_size,
                                dtype=np.int32,
                                name="labels")
    logits = keras.layers.Input((max_time, num_classes),
                                batch_size=batch_size,
                                dtype=np.float32,
                                name="logits")
    label_length = keras.layers.Input((),
                                      batch_size=batch_size,
                                      dtype=np.int32,
                                      name="label_length")
    logit_length = keras.layers.Input((),
                                      batch_size=batch_size,
                                      dtype=np.int32,
                                      name="logit_length")
    x = logits

    dense_layer = layers.Dense(num_classes,
                               kernel_initializer='ones',
                               bias_initializer='ones')
    x = dense_layer(x)

    transpose_fn = lambda x: keras.backend.permute_dimensions(x, (1, 0, 2))
    transpose_layer = layers.Lambda(transpose_fn)
    x = transpose_layer(x)

    if log_softmax:
      log_softmax_fn = lambda x: nn_ops.log_softmax_v2(x, axis=2)
      log_softmax_layer = layers.Lambda(log_softmax_fn)
      x = log_softmax_layer(x)

    loss = loss_layer(labels, x, label_length, logit_length)

    m = keras.Model(inputs=[labels, logits, label_length, logit_length],
                    outputs=loss)

    def get_loss_output(_, y_pred):
      return y_pred

    m.compile('sgd', loss=get_loss_output)
    return m

  @staticmethod
  def create_loss_endpoint_model(endpoint_layer, ctc_params,
                                 log_softmax=False):
    batch_size = ctc_params["batch_size"]
    num_classes = ctc_params["num_classes"]
    max_label_length = ctc_params["max_label_length"]
    max_time = ctc_params["max_time"]

    labels = keras.layers.Input((max_label_length),
                                batch_size=batch_size,
                                dtype=np.int32,
                                name="labels")
    logits = keras.layers.Input((max_time, num_classes),
                                batch_size=batch_size,
                                dtype=np.float32,
                                name="logits")
    label_length = keras.layers.Input((),
                                      batch_size=batch_size,
                                      dtype=np.int32,
                                      name="label_length")
    logit_length = keras.layers.Input((),
                                      batch_size=batch_size,
                                      dtype=np.int32,
                                      name="logit_length")
    x = logits

    dense_layer = layers.Dense(num_classes,
                               kernel_initializer='ones',
                               bias_initializer='ones')
    x = dense_layer(x)

    transpose_fn = lambda x: keras.backend.permute_dimensions(x, (1, 0, 2))
    transpose_layer = layers.Lambda(transpose_fn)
    x = transpose_layer(x)

    if log_softmax:
      log_softmax_fn = lambda x: nn_ops.log_softmax_v2(x, axis=2)
      log_softmax_layer = layers.Lambda(log_softmax_fn)
      x = log_softmax_layer(x)

    x = endpoint_layer(x,
                       logit_length,
                       labels=labels,
                       label_length=label_length)

    m = keras.Model(inputs=[labels, logits, label_length, logit_length],
                    outputs=x)
    m.compile('sgd')
    return m

  @staticmethod
  def create_dataset(ctc_params):
    batch_size = ctc_params["batch_size"]
    num_classes = ctc_params["num_classes"]
    max_label_length = ctc_params["max_label_length"]
    max_time = ctc_params["max_time"]
    actual_label_length = ctc_params["actual_label_length"]
    blank_index = ctc_params["blank_index"]

    def data_generator():
      np.random.seed(9)

      while True:
        labels = (np.random.randint(1, num_classes, size=[max_label_length]) \
                     + blank_index) % num_classes
        logits = np.float32(
            np.random.randint(0, num_classes, size=[max_time, num_classes]))
        label_length = np.int32(actual_label_length)
        logit_length = np.int32(max_time)
        yield ((labels, logits, label_length, logit_length), labels)

    output_types = ((np.int32, np.float32, np.int32, np.int32), np.int32)
    output_shapes = (((max_label_length), (max_time, num_classes), (), ()),
                     (max_label_length))

    dataset = Dataset.from_generator(data_generator, output_types,
                                     output_shapes)
    dataset = dataset.take(batch_size).cache()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

  @test_util.run_v2_only
  def testCTCLossEvaluate(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_loss_endpoint_model(loss_layer_cpu, ctc_params)
    loss_cpu = model_cpu.evaluate(dataset, steps=1)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"])
      model_ipu = self.create_loss_model(loss_layer_ipu,
                                         ctc_params,
                                         log_softmax=True)
      loss_ipu = model_ipu.evaluate(dataset, steps=1)

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)

  @test_util.run_v2_only
  def testCTCLossFromLogitsEvaluate(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_loss_endpoint_model(loss_layer_cpu, ctc_params)
    loss_cpu = model_cpu.evaluate(dataset, steps=1)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"],
                                         from_logits=True)
      model_ipu = self.create_loss_model(loss_layer_ipu, ctc_params)
      loss_ipu = model_ipu.evaluate(dataset, steps=1)

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)

  @test_util.run_v2_only
  def testCTCLossFit(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_loss_endpoint_model(loss_layer_cpu, ctc_params)
    history_cpu = model_cpu.fit(dataset, steps_per_epoch=1)
    loss_cpu = history_cpu.history["loss"]

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"])
      model_ipu = self.create_loss_model(loss_layer_ipu,
                                         ctc_params,
                                         log_softmax=True)
      history_ipu = model_ipu.fit(dataset, steps_per_epoch=1)
      loss_ipu = history_ipu.history["loss"]

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)
    self.assertAllClose(model_cpu.get_weights(), model_ipu.get_weights())

  @test_util.run_v2_only
  def testCTCLossFromLogitsFit(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_loss_endpoint_model(loss_layer_cpu, ctc_params)
    history_cpu = model_cpu.fit(dataset, steps_per_epoch=1)
    loss_cpu = history_cpu.history["loss"]

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"],
                                         from_logits=True)
      model_ipu = self.create_loss_model(loss_layer_ipu, ctc_params)
      history_ipu = model_ipu.fit(dataset, steps_per_epoch=1)
      loss_ipu = history_ipu.history["loss"]

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)
    self.assertAllClose(model_cpu.get_weights(), model_ipu.get_weights())

  @parameterized.named_parameters(generate_beam_search_params())
  @test_util.run_v2_only
  def testCTCPredictions(self, params):
    ctc_params = params.params
    beam_width = ctc_params["beam_width"]
    top_paths = ctc_params["top_paths"]
    blank_index = ctc_params[
        "num_classes"] - 1  # cpu version has preset blank index

    np.random.seed(ctc_params["seed"])
    inputs = np.random.rand(ctc_params["batch_size"] * 20,
                            ctc_params["max_time"], ctc_params["num_classes"])
    input_lengths = np.full([ctc_params["batch_size"] * 20],
                            ctc_params["max_time"] - 1,
                            dtype=np.int32)

    # CPU model
    predictions_layer_cpu = CTCPredictionsCpu(blank_index=blank_index,
                                              beam_width=beam_width,
                                              top_paths=top_paths)
    model_cpu = self.create_predictions_model(predictions_layer_cpu,
                                              ctc_params)

    cpu_predictions = model_cpu.predict(x=[inputs, input_lengths],
                                        batch_size=ctc_params["batch_size"])

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # IPU model
      predictions_layer_ipu = ipu.keras.layers.CTCPredictionsLayer(
          blank_index=blank_index,
          from_logits=True,
          beam_width=beam_width,
          top_paths=top_paths)
      model_ipu = self.create_predictions_model(predictions_layer_ipu,
                                                ctc_params)

      ipu_predictions = model_ipu.predict(x=[inputs, input_lengths],
                                          batch_size=ctc_params["batch_size"])

    self.assertAllClose(cpu_predictions, ipu_predictions)

  @test_util.run_v2_only
  def testCTCInferenceLayerGetConfig(self):
    layer = ipu.keras.layers.CTCInferenceLayer(blank_index=0,
                                               from_logits=True,
                                               beam_width=10,
                                               top_paths=1)
    config = layer.get_config()
    layer2 = ipu.keras.layers.CTCInferenceLayer.from_config(config)
    self.assertEqual(config, layer2.get_config())

  @test_util.run_v2_only
  def testCTCPredictionsLayerGetConfig(self):
    layer = ipu.keras.layers.CTCPredictionsLayer(blank_index=0,
                                                 beam_width=10,
                                                 top_paths=1)
    config = layer.get_config()
    layer2 = ipu.keras.layers.CTCPredictionsLayer.from_config(config)
    self.assertEqual(config, layer2.get_config())

  @test_util.run_v2_only
  def testCTCLossGetConfig(self):
    layer = ipu.keras.CTCLoss(blank_index=0)
    config = layer.get_config()
    layer2 = ipu.keras.CTCLoss.from_config(config)
    self.assertEqual(config, layer2.get_config())


if __name__ == '__main__':
  test.main()

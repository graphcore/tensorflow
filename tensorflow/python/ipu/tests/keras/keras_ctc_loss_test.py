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

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data import Dataset
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import ctc_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

dataType = np.float32


class CTCLossEndpointCpu(layers.Layer):
  def __init__(self, blank_index=0, name=None):
    super().__init__(name=name)
    self.blank_index = blank_index

  def call(self, labels, logits, label_length, logit_length, **kwargs):  # pylint: disable=W0221
    loss = ctc_ops.ctc_loss_v2(labels,
                               logits,
                               label_length,
                               logit_length,
                               blank_index=self.blank_index)
    loss = math_ops.reduce_mean(loss)
    self.add_loss(loss)
    return logits


class CTCLossTest(test.TestCase):
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
  def create_model(loss_layer, model, ctc_params, log_softmax=False):
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

    m = model(inputs=[labels, logits, label_length, logit_length],
              outputs=loss)

    def get_loss_output(_, y_pred):
      return y_pred

    m.compile('sgd', loss=get_loss_output)
    return m

  @staticmethod
  def create_endpoint_model(endpoint_layer,
                            model,
                            ctc_params,
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

    x = endpoint_layer(labels, x, label_length, logit_length)

    m = model(inputs=[labels, logits, label_length, logit_length], outputs=x)
    m.compile('sgd')
    return m

  @staticmethod
  def create_dataset(ctc_params):
    batch_size = ctc_params["batch_size"]
    num_classes = ctc_params["num_classes"]
    max_label_length = ctc_params["max_label_length"]
    max_time = ctc_params["max_time"]
    actual_label_length = ctc_params["actual_label_length"]

    def data_generator():
      while True:
        labels = np.random.randint(1, num_classes, size=[max_label_length])
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
    model_cpu = self.create_endpoint_model(loss_layer_cpu, keras.Model,
                                           ctc_params)
    loss_cpu = model_cpu.evaluate(dataset, steps=1)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"])
      model_ipu = self.create_model(loss_layer_ipu,
                                    ipu.keras.Model,
                                    ctc_params,
                                    log_softmax=True)
      loss_ipu = model_ipu.evaluate(dataset, steps=1)[0]

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)

  @test_util.run_v2_only
  def testCTCLossFromLogitsEvaluate(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_endpoint_model(loss_layer_cpu, keras.Model,
                                           ctc_params)
    loss_cpu = model_cpu.evaluate(dataset, steps=1)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"],
                                         from_logits=True)
      model_ipu = self.create_model(loss_layer_ipu, ipu.keras.Model,
                                    ctc_params)
      loss_ipu = model_ipu.evaluate(dataset, steps=1)[0]

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)

  @test_util.run_v2_only
  def testCTCLossFit(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_endpoint_model(loss_layer_cpu, keras.Model,
                                           ctc_params)
    history_cpu = model_cpu.fit(dataset, steps_per_epoch=1)
    loss_cpu = history_cpu.history["loss"]

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"])
      model_ipu = self.create_model(loss_layer_ipu,
                                    ipu.keras.Model,
                                    ctc_params,
                                    log_softmax=True)
      history_ipu = model_ipu.fit(dataset, steps_per_epoch=1)
      loss_ipu = history_ipu.history["loss"]
      print(history_ipu)

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)
    self.assertAllClose(model_cpu.get_weights(), model_ipu.get_weights())

  @test_util.run_v2_only
  def testCTCLossFromLogitsFit(self):
    ctc_params = self.get_params()
    dataset = self.create_dataset(ctc_params)

    # CPU model
    loss_layer_cpu = CTCLossEndpointCpu(blank_index=ctc_params["blank_index"])
    model_cpu = self.create_endpoint_model(loss_layer_cpu, keras.Model,
                                           ctc_params)
    history_cpu = model_cpu.fit(dataset, steps_per_epoch=1)
    loss_cpu = history_cpu.history["loss"]

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # IPU model
      loss_layer_ipu = ipu.keras.CTCLoss(blank_index=ctc_params["blank_index"],
                                         from_logits=True)
      model_ipu = self.create_model(loss_layer_ipu, ipu.keras.Model,
                                    ctc_params)
      history_ipu = model_ipu.fit(dataset, steps_per_epoch=1)
      loss_ipu = history_ipu.history["loss"]
      print(history_ipu)

    self.assertEqual(np.size(loss_ipu), 1)
    self.assertAllClose(loss_cpu, loss_ipu)
    self.assertAllClose(model_cpu.get_weights(), model_ipu.get_weights())


if __name__ == '__main__':
  test.main()

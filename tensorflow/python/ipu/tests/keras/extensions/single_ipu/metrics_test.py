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
# ==============================================================================
"""Tests for Keras metrics functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python import ipu


def _get_model(compile_metrics):
  model_layers = [
      layers.Dense(3, activation='relu', kernel_initializer='ones'),
      layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
  ]

  model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
  model.compile(loss='mae',
                metrics=compile_metrics,
                optimizer='rmsprop',
                run_eagerly=testing_utils.should_run_eagerly())
  return model


@keras_parameterized.run_with_all_model_types()
@keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                         always_skip_v1=True)
class ResetStatesTest(keras_parameterized.TestCase):
  def setUp(self):
    super(ResetStatesTest, self).setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super(ResetStatesTest, self).tearDown()

  def test_reset_states_false_positives(self):
    fp_obj = metrics.FalsePositives()
    model = _get_model([fp_obj])
    x = np.ones((128, 4))
    y = np.zeros((128, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 128.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 128.)

  def test_reset_states_false_negatives(self):
    fn_obj = metrics.FalseNegatives()
    model = _get_model([fn_obj])
    x = np.zeros((128, 4))
    y = np.ones((128, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 128.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 128.)

  def test_reset_states_true_negatives(self):
    tn_obj = metrics.TrueNegatives()
    model = _get_model([tn_obj])
    x = np.zeros((128, 4))
    y = np.zeros((128, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 128.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 128.)

  def test_reset_states_true_positives(self):
    tp_obj = metrics.TruePositives()
    model = _get_model([tp_obj])
    x = np.ones((128, 4))
    y = np.ones((128, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 128.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 128.)

  def test_reset_states_precision(self):
    p_obj = metrics.Precision()
    model = _get_model([p_obj])
    x = np.concatenate((np.ones((64, 4)), np.ones((64, 4))))
    y = np.concatenate((np.ones((64, 1)), np.zeros((64, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 64.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 64.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 64.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 64.)

  def test_reset_states_recall(self):
    r_obj = metrics.Recall()
    model = _get_model([r_obj])
    x = np.concatenate((np.ones((64, 4)), np.zeros((64, 4))))
    y = np.concatenate((np.ones((64, 1)), np.ones((64, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 64.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 64.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 64.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 64.)

  def test_reset_states_auc(self):
    auc_obj = metrics.AUC(num_thresholds=3)
    model = _get_model([auc_obj])
    x = np.concatenate((np.ones((32, 4)), np.zeros((32, 4)), np.zeros(
        (32, 4)), np.ones((32, 4))))
    y = np.concatenate((np.ones((32, 1)), np.zeros((32, 1)), np.ones(
        (32, 1)), np.zeros((32, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 32.)

  def test_reset_states_auc_manual_thresholds(self):
    auc_obj = metrics.AUC(thresholds=[0.5])
    model = _get_model([auc_obj])
    x = np.concatenate((np.ones((32, 4)), np.zeros((32, 4)), np.zeros(
        (32, 4)), np.ones((32, 4))))
    y = np.concatenate((np.ones((32, 1)), np.zeros((32, 1)), np.ones(
        (32, 1)), np.zeros((32, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 32.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 32.)


if __name__ == '__main__':
  test.main()

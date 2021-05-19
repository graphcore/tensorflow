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
""" Tests replacement of Keras layers with IPU Keras layers """

from absl.testing import parameterized
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.python import ones
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import layers as ipu_layers
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.keras import Sequential
from tensorflow.python.ipu.keras.layer_replacement import _LayerInstanceDescriptor
from tensorflow.python.ipu.keras.layer_replacement import _LayerClassDescriptor
from tensorflow.python.ipu.keras.layer_replacement import IPULayerReplacer
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.platform import test

# This is the list of known ipu.keras.Layer classes. If a new IPU Keras layer
# is created but not added here, then the tests in this module will fail.
_ipu_layers = [
    "Dropout",
    "Embedding",
    "GroupNorm",
    "InstanceNorm",
    "LayerNormalization",
    "LayerNorm",  # alias
    "PopnnLSTM",
    "LSTM",  # alias
    "PopnnGRU",
    "GRU",  # alias
    "SerialDense"
]

# These layers are in the TF additions repo, so we don't support
# their substitution with ipu.keras layers.
_ipu_blacklisted_layers = ["GroupNorm", "InstanceNorm", "SerialDense"]

# Per class test cases for IPULayerReplacer.
_test_cases = [{
    'testcase_name': 'Dropout',
    'keras_type': layers.Dropout,
    'ipu_keras_type': ipu_layers.Dropout,
    'init_args': (0.5,),
    'init_kwargs': {
        'seed': 123
    },
    'call_args': (ones(16, 10),),
    'call_kwargs': {}
}, {
    'testcase_name': 'Embedding',
    'keras_type': layers.Embedding,
    'ipu_keras_type': ipu_layers.Embedding,
    'init_args': (10, 2),
    'init_kwargs': {},
    'call_args': (ones((1, 10), dtype=dtypes.int32),),
    'call_kwargs': {}
}, {
    'testcase_name': 'LayerNormalization',
    'keras_type': layers.LayerNormalization,
    'ipu_keras_type': ipu_layers.LayerNorm,
    'init_args': (),
    'init_kwargs': {},
    'call_args': (ones((16, 10)),),
    'call_kwargs': {}
}, {
    'testcase_name': 'LSTM',
    'keras_type': layers.LSTM,
    'ipu_keras_type': ipu_layers.LSTM,
    'init_args': (5,),
    'init_kwargs': {},
    'call_args': (ones((8, 100, 10)),),
    'call_kwargs': {}
}, {
    'testcase_name': 'GRU',
    'keras_type': layers.GRU,
    'ipu_keras_type': ipu_layers.GRU,
    'init_args': (5,),
    'init_kwargs': {
        'implementation': 1
    },
    'call_args': (ones((8, 100, 10)),),
    'call_kwargs': {}
}]

# Per class model training and prediction test cases.
_train_pred_test_cases = [
    {
        'testcase_name':
        'Dropout',
        'model_fn':
        lambda: [
            layers.Dense(10, kernel_initializer=initializers.Constant(0.1)),
            layers.Dropout(0.01),
            layers.Dense(2, kernel_initializer=initializers.Constant(0.1)),
            layers.Dropout(0.99),
            layers.ReLU()
        ],
        'features': (ones((16, 10)),),
        'targets': (ones((16, 2)))
    },
    {
        'testcase_name':
        'Embedding',
        'model_fn':
        lambda: [
            layers.Embedding(
                10, 2, embeddings_initializer=initializers.Constant(0.1)),
            layers.Dense(2, kernel_initializer=initializers.Constant(0.1)),
        ],
        'features': (ones((16, 10), dtype=dtypes.int32),),
        'targets': (ones((16, 2), dtype=dtypes.int32))
    },
    {
        'testcase_name':
        'LayerNormalization',
        'model_fn':
        lambda: [
            layers.Dense(10, kernel_initializer=initializers.Constant(0.1)),
            layers.LayerNormalization(),
            layers.Dense(2, kernel_initializer=initializers.Constant(0.1)),
        ],
        'features': (ones((16, 10)),),
        'targets': (ones((16, 2)))
    },
    {
        'testcase_name':
        'LSTM',
        'model_fn':
        lambda: [
            layers.LSTM(10,
                        kernel_initializer=initializers.Constant(0.1),
                        recurrent_initializer=initializers.Constant(0.1))
        ],
        'features': (ones((8, 100, 10)),),
        'targets': (ones((8, 10)))
    },
    {
        'testcase_name':
        'GRU',
        'model_fn':
        lambda: [
            layers.GRU(10,
                       kernel_initializer=initializers.Constant(0.1),
                       recurrent_initializer=initializers.Constant(0.1))
        ],
        'features': (ones((8, 100, 10)),),
        'targets': (ones((8, 10)))
    },
]


class KerasLayerReplacementTest(test.TestCase, parameterized.TestCase):
  @test_util.run_v2_only
  def testCanFindLayers(self):
    # Check that all IPU keras layers can be found.
    replacer = IPULayerReplacer()
    found_layers = replacer.get_ipu_layer_names()

    self.assertEqual(len(found_layers), len(_ipu_layers))

    _ipu_layers.sort()
    found_layers.sort()
    self.assertEqual(found_layers, _ipu_layers)

  @test_util.run_v2_only
  def testBlacklist(self):
    replacer = IPULayerReplacer()
    blacklist = replacer.get_blacklisted_ipu_layer_names()
    self.assertEqual(blacklist, _ipu_blacklisted_layers)

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testVerifyTestCase(self, keras_type, ipu_keras_type, init_args,
                         init_kwargs, call_args, call_kwargs):
    self.assertNotEqual(keras_type, ipu_keras_type)
    self.assertTrue(issubclass(keras_type, layers.Layer))
    self.assertTrue(issubclass(ipu_keras_type, layers.Layer))

    self.assertTrue(isinstance(init_args, tuple))
    self.assertTrue(isinstance(init_kwargs, dict))
    self.assertTrue(isinstance(call_args, tuple))
    self.assertTrue(isinstance(call_kwargs, dict))

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testCanFindInitArgs(
      self,
      keras_type,
      ipu_keras_type,
      init_args,  # pylint: disable=unused-argument
      init_kwargs,
      call_args,  # pylint: disable=unused-argument
      call_kwargs):  # pylint: disable=unused-argument
    # An assert will fail if there is an issue.
    keras_desc = _LayerInstanceDescriptor(keras_type(*init_args,
                                                     **init_kwargs))
    ipu_keras_desc = _LayerInstanceDescriptor(
        ipu_keras_type(*init_args, **init_kwargs))

    # Check that the configurations match.
    self.assertEqual(keras_desc.get_init_args(),
                     ipu_keras_desc.get_init_args())

    self.assertTrue(all(k in ipu_keras_desc.get_init_kwargs()\
      for k in keras_desc.get_init_kwargs()))

    self.assertEqual(keras_desc.get_init_kwargs(),
                     {k: ipu_keras_desc.get_init_kwargs()[k]\
                       for k in keras_desc.get_init_kwargs().keys()})

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testCanFindCallArgs(
      self,
      keras_type,
      ipu_keras_type,
      init_args,  # pylint: disable=unused-argument
      init_kwargs,  # pylint: disable=unused-argument
      call_args,  # pylint: disable=unused-argument
      call_kwargs):  # pylint: disable=unused-argument
    keras_desc = _LayerClassDescriptor(keras_type)
    ipu_keras_desc = _LayerClassDescriptor(ipu_keras_type)

    self.assertEqual(keras_desc.get_call_args(),
                     ipu_keras_desc.get_call_args())

    self.assertTrue(all(k in ipu_keras_desc.get_call_kwargs()\
      for k in keras_desc.get_call_kwargs()))

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testCanReplaceLayer(
      self,
      keras_type,
      ipu_keras_type,
      init_args,  # pylint: disable=unused-argument
      init_kwargs,
      call_args,  # pylint: disable=unused-argument
      call_kwargs):  # pylint: disable=unused-argument
    keras_layer = keras_type(*init_args, **init_kwargs)

    replacer = IPULayerReplacer()
    new_layer = replacer(keras_layer)

    self.assertEqual(type(new_layer), ipu_keras_type)

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testCanCallReplacedLayer(
      self,
      keras_type,
      ipu_keras_type,  # pylint: disable=unused-argument
      init_args,  # pylint: disable=unused-argument
      init_kwargs,
      call_args,
      call_kwargs):
    keras_layer = keras_type(*init_args, **init_kwargs)

    replacer = IPULayerReplacer()
    new_layer = replacer(keras_layer)

    new_layer(*call_args, **call_kwargs)

  @parameterized.named_parameters(*_test_cases)
  @test_util.run_v2_only
  def testReplacedLayerWeightsSame(
      self,
      keras_type,
      ipu_keras_type,  # pylint: disable=unused-argument
      init_args,  # pylint: disable=unused-argument
      init_kwargs,
      call_args,  # pylint: disable=unused-argument
      call_kwargs):  # pylint: disable=unused-argument
    keras_layer = keras_type(*init_args, **init_kwargs)

    replacer = IPULayerReplacer()
    new_layer = replacer(keras_layer)

    for x, y in zip(keras_layer.trainable_weights,
                    new_layer.trainable_weights):
      self.assertAllEqual(x, y)

  @parameterized.named_parameters(*_train_pred_test_cases)
  @test_util.run_v2_only
  def testTrainingConsistency(self, model_fn, features, targets):
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Configure an IPU.
      cfg = IPUConfig()
      cfg._profiling.profiling = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # Build model without layer replacement.
      model = Sequential(model_fn())
      model.compile('sgd', 'mse')
      res = model.fit(features, targets, batch_size=1, epochs=3)

      # Build model with layer replacement.
      replacer = IPULayerReplacer()
      seq_model_replaced = []
      for layer in model_fn():
        seq_model_replaced.append(replacer(layer))

      model_replaced = Sequential(seq_model_replaced)
      model_replaced.compile('sgd', 'mse')
      res_replaced = model_replaced.fit(features,
                                        targets,
                                        batch_size=1,
                                        epochs=3)

      self.assertAllClose(res.history['loss'], res_replaced.history['loss'])

  @parameterized.named_parameters(*_train_pred_test_cases)
  @test_util.run_v2_only
  def testPredictionConsistency(self, model_fn, features, targets):
    pass


if __name__ == '__main__':
  test.main()

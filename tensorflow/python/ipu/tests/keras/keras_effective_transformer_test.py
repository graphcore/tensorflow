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
"""Test for IPU Keras Effective Transformer layers."""

import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ipu.keras.layers.effective_transformer import EffectiveTransformer
from tensorflow.python.platform import googletest


def _generate_sequences(seq_lens, max_len, dtype):
  sequences = np.zeros((seq_lens.shape[0], max_len), dtype=dtype)
  for i, n in enumerate(np.squeeze(seq_lens)):
    sequences[i, 0:n] = i + 1

  return sequences


def _generate_random_binary_mask(a, b):
  mask = np.random.randint(2, size=(a, b))
  return mask.astype(np.bool)


test_seq_lens = np.expand_dims(np.array([8, 3, 5, 7, 3], dtype=np.int32), 1)
test_seq_lens_2 = np.expand_dims(np.array([2, 8, 11, 3, 9], dtype=np.int32), 1)

TEST_CASES = [{
    'testcase_name': "TestCase1",
    'output_layer_size': 12,
    'max_batch_size': 10,
    'from_seq_lens': test_seq_lens,
    'to_seq_lens': test_seq_lens_2,
    'from_row_len': np.amax(test_seq_lens),
    'to_row_len': np.amax(test_seq_lens_2),
    'dtype': np.float32,
    'seq_per_iter': 2,
    'use_scale': False,
    'q_mask': None,
    'attention_heads': 4,
    'attention_head_size': 16,
}, {
    'testcase_name': "TestCase2",
    'output_layer_size': 12,
    'max_batch_size': 10,
    'from_seq_lens': test_seq_lens,
    'to_seq_lens': test_seq_lens_2,
    'from_row_len': np.amax(test_seq_lens),
    'to_row_len': np.amax(test_seq_lens_2),
    'dtype': np.float32,
    'seq_per_iter': 2,
    'use_scale': True,
    'q_mask': None,
    'attention_heads': 4,
    'attention_head_size': 16,
}, {
    'testcase_name': "TestCase3",
    'output_layer_size': 12,
    'max_batch_size': 10,
    'from_seq_lens': test_seq_lens,
    'to_seq_lens': test_seq_lens_2,
    'from_row_len': np.amax(test_seq_lens),
    'to_row_len': np.amax(test_seq_lens_2),
    'dtype': np.float32,
    'seq_per_iter': 2,
    'use_scale': True,
    'q_mask': _generate_random_binary_mask(len(test_seq_lens), 4),
    'attention_heads': 4,
    'attention_head_size': 16
}]

SEQUENCE_PADDING_TEST_CASES = [{
    'testcase_name':
    'TestCase1',
    'padded_sequence':
    np.array([[1, 1, 0, 0], [2, 2, 2, 0], [3, 0, 0, 0], [4, 4, 0, 0]],
             dtype=np.float32),
    'unpadded_sequence':
    np.array([[1, 1, 2, 2], [2, 3, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=np.float32),
    'sequence_lengths':
    np.array([2, 3, 1, 2], dtype=np.int32)
}, {
    'testcase_name':
    'TestCase2',
    'padded_sequence':
    np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             dtype=np.float32),
    'unpadded_sequence':
    np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             dtype=np.float32),
    'sequence_lengths':
    np.array([4, 4, 4, 4], dtype=np.int32)
}, {
    'testcase_name':
    'TestCase3',
    'padded_sequence':
    np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]],
             dtype=np.float32),
    'unpadded_sequence':
    np.array([[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=np.float32),
    'sequence_lengths':
    np.array([1, 1, 1, 1], dtype=np.int32)
}]


class IPUEffectiveTransformerLayerTest(test_util.TensorFlowTestCase,
                                       parameterized.TestCase):
  def testMismatchedSequenceRanks(self):
    from_seq = np.zeros((3, 3, 1), dtype=np.float32)
    to_seq = np.zeros((3, 3), dtype=np.float32)

    seq_lens = np.array([3] * 3, dtype=np.int32)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(ValueError, r"must be equal rank"):
      transformer([from_seq, seq_lens, to_seq, seq_lens])

  def testMismatchedSequenceCount(self):
    from_seq = np.zeros((4, 3), dtype=np.float32)
    from_seq_lens = np.array([3] * 4, dtype=np.int32)

    to_seq = np.zeros((3, 3), dtype=np.float32)
    to_seq_lens = np.array([3] * 3, dtype=np.int32)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(ValueError,
                                r"must contain an equal number of sequences"):
      transformer([from_seq, from_seq_lens, to_seq, to_seq_lens])

  def testMismatchedSequenceLen(self):
    from_seq = np.zeros((3, 3), dtype=np.float32)
    to_seq = np.zeros((3, 3), dtype=np.float32)

    seq_lens = np.array([3] * 4, dtype=np.int32)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(
        ValueError, r"have lengths equal to the number of sequences provided"):
      transformer([from_seq, seq_lens, to_seq, seq_lens])

  def testMismatchedSequenceLen2(self):
    from_seq = np.zeros((3, 3), dtype=np.float32)
    from_seq_lens = np.array([3] * 3, dtype=np.int32)

    to_seq = np.zeros((3, 3), dtype=np.float32)
    to_seq_lens = np.array([3] * 2, dtype=np.int32)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(
        ValueError, r"have lengths equal to the number of sequences provided"):
      transformer([from_seq, from_seq_lens, to_seq, to_seq_lens])

  def testMismatchedDtypes(self):
    from_seq = np.zeros((3, 3), dtype=np.float16)
    to_seq = np.zeros((3, 3), dtype=np.int32)
    seq_lens = np.array([3] * 3, dtype=np.int32)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(
        ValueError, r"from_sequences and to_sequences must be equal types"):
      transformer([from_seq, seq_lens, to_seq, seq_lens])

  def testIncorrectQMask(self):
    from_seq = np.zeros((3, 3), dtype=np.float32)
    to_seq = np.zeros_like(from_seq)
    seq_lens = np.array([3] * 3, dtype=np.int32)

    # Should be [num_sequences, num_heads]
    q_mask = np.zeros((3, 2), dtype=np.bool)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(ValueError, r"q_mask must have shape"):
      transformer([from_seq, seq_lens, to_seq, seq_lens, q_mask])

  def testIncorrectQMask2(self):
    from_seq = np.zeros((3, 3), dtype=np.float32)
    to_seq = np.zeros_like(from_seq)
    seq_lens = np.zeros([3] * 3, dtype=np.int32)

    # Should be [num_sequences, num_heads]
    q_mask = np.zeros((4, 1), dtype=np.bool)

    transformer = EffectiveTransformer(12, 10)
    with self.assertRaisesRegex(ValueError, r"q_mask must have shape"):
      transformer([from_seq, seq_lens, to_seq, seq_lens, q_mask])

  @parameterized.named_parameters(*TEST_CASES)
  def testInference(self, output_layer_size, max_batch_size, from_seq_lens,
                    to_seq_lens, from_row_len, to_row_len, dtype, seq_per_iter,
                    use_scale, q_mask, attention_heads, attention_head_size):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # Generate some data.
      from_sequences = _generate_sequences(from_seq_lens, from_row_len, dtype)
      to_sequences = _generate_sequences(to_seq_lens, to_row_len, dtype)

      input_tensors = [
          from_sequences, from_seq_lens, to_sequences, to_seq_lens
      ]

      if not q_mask is None:
        input_tensors.append(q_mask)

      transformer_kwargs = {
          'output_layer_size': output_layer_size,
          'max_batch_size': max_batch_size,
          'use_scale': use_scale,
          'num_attention_heads': attention_heads,
          'attention_head_size': attention_head_size,
          'sequences_per_iter': seq_per_iter,
          'embedding_initializer': 'ones',
          'output_initializer': 'ones'
      }

      # Build an Effective Transformer.
      transformer = EffectiveTransformer(**transformer_kwargs)

      @def_function.function(experimental_compile=True)
      def f(inputs):
        return transformer(inputs)

      res = strategy.run(f, args=[input_tensors])

      # Check output shapes.
      self.assertEqual(res[0].shape,
                       (from_sequences.shape[0], output_layer_size))

      self.assertEqual(res[1].shape,
                       (from_sequences.shape[0], from_sequences.shape[1],
                        attention_heads, attention_head_size))

  @parameterized.named_parameters(*TEST_CASES)
  def testInferenceKeras(self, output_layer_size, max_batch_size,
                         from_seq_lens, to_seq_lens, from_row_len, to_row_len,
                         dtype, seq_per_iter, use_scale, q_mask,
                         attention_heads, attention_head_size):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # Generate some data.
      from_sequences = _generate_sequences(from_seq_lens, from_row_len, dtype)
      to_sequences = _generate_sequences(to_seq_lens, to_row_len, dtype)

      # Input layers.
      from_seq = keras.layers.Input(shape=(from_sequences.shape[1]),
                                    batch_size=from_sequences.shape[0],
                                    dtype=dtype)

      from_seq_len = keras.layers.Input(shape=(1),
                                        batch_size=from_seq_lens.shape[0],
                                        dtype=np.int32)

      to_seq = keras.layers.Input(shape=(to_sequences.shape[1]),
                                  batch_size=to_sequences.shape[0],
                                  dtype=dtype)

      to_seq_len = keras.layers.Input(shape=(1),
                                      batch_size=to_seq_lens.shape[0],
                                      dtype=np.int32)

      input_layers = [from_seq, from_seq_len, to_seq, to_seq_len]
      input_tensors = [
          from_sequences, from_seq_lens, to_sequences, to_seq_lens
      ]

      if not q_mask is None:
        mask = keras.layers.Input(shape=(q_mask.shape[1]),
                                  batch_size=q_mask.shape[0],
                                  dtype=q_mask.dtype)
        input_layers.append(mask)
        input_tensors.append(q_mask)

      transformer_kwargs = {
          'output_layer_size': output_layer_size,
          'max_batch_size': max_batch_size,
          'use_scale': use_scale,
          'num_attention_heads': attention_heads,
          'attention_head_size': attention_head_size,
          'sequences_per_iter': seq_per_iter,
          'embedding_initializer': 'ones',
          'output_initializer': 'ones'
      }

      # Build an Effective Transformer.
      transformer = EffectiveTransformer(**transformer_kwargs)
      x = transformer(input_layers)

      model = keras.Model(inputs=input_layers, outputs=x)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      model.compile(opt, loss='mse')

      res = model.predict(input_tensors, batch_size=from_sequences.shape[0])

      # Check output shapes.
      self.assertEqual(res[0].shape,
                       (from_sequences.shape[0], output_layer_size))

      self.assertEqual(res[1].shape,
                       (from_sequences.shape[0], from_sequences.shape[1],
                        attention_heads, attention_head_size))

  @parameterized.named_parameters(*TEST_CASES)
  def testTraining(self, output_layer_size, max_batch_size, from_seq_lens,
                   to_seq_lens, from_row_len, to_row_len, dtype, seq_per_iter,
                   use_scale, q_mask, attention_heads, attention_head_size):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # Generate some data.
      from_sequences = _generate_sequences(from_seq_lens, from_row_len, dtype)
      to_sequences = _generate_sequences(to_seq_lens, to_row_len, dtype)

      input_tensors = [
          from_sequences, from_seq_lens, to_sequences, to_seq_lens
      ]

      if not q_mask is None:
        input_tensors.append(q_mask)

      transformer_kwargs = {
          'output_layer_size': output_layer_size,
          'max_batch_size': max_batch_size,
          'use_scale': use_scale,
          'num_attention_heads': attention_heads,
          'attention_head_size': attention_head_size,
          'sequences_per_iter': seq_per_iter,
          'embedding_initializer': 'ones',
          'output_initializer': 'ones'
      }

      # Build an Effective Transformer.
      transformer = EffectiveTransformer(**transformer_kwargs)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.01)

      @def_function.function(experimental_compile=True)
      def f(inputs, targets):
        with GradientTape() as tape:
          y = transformer(inputs)[0]
          l = keras.losses.mean_squared_error(targets, y)

          g = tape.gradient(l, transformer.trainable_variables)
          v = transformer.trainable_variables
          opt.apply_gradients(zip(g, v))
          return l

      targets = np.ones((from_sequences.shape[0], output_layer_size),
                        dtype=dtype)

      # Check losses are decreasing monotonically.
      last_loss = float('inf')
      for _ in range(3):
        l = strategy.run(f, [input_tensors, targets])[0]
        self.assertLess(l, last_loss)
        last_loss = l

  @parameterized.named_parameters(*TEST_CASES)
  def testTrainingKeras(self, output_layer_size, max_batch_size, from_seq_lens,
                        to_seq_lens, from_row_len, to_row_len, dtype,
                        seq_per_iter, use_scale, q_mask, attention_heads,
                        attention_head_size):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # Generate some data.
      from_sequences = _generate_sequences(from_seq_lens, from_row_len, dtype)
      to_sequences = _generate_sequences(to_seq_lens, to_row_len, dtype)

      # Input layers.
      from_seq = keras.layers.Input(shape=(from_sequences.shape[1]),
                                    batch_size=from_sequences.shape[0],
                                    dtype=dtype)

      from_seq_len = keras.layers.Input(shape=(1),
                                        batch_size=from_seq_lens.shape[0],
                                        dtype=np.int32)

      to_seq = keras.layers.Input(shape=(to_sequences.shape[1]),
                                  batch_size=to_sequences.shape[0],
                                  dtype=dtype)

      to_seq_len = keras.layers.Input(shape=(1),
                                      batch_size=to_seq_lens.shape[0],
                                      dtype=np.int32)

      input_layers = [from_seq, from_seq_len, to_seq, to_seq_len]
      input_tensors = [
          from_sequences, from_seq_lens, to_sequences, to_seq_lens
      ]

      if not q_mask is None:
        mask = keras.layers.Input(shape=(q_mask.shape[1]),
                                  batch_size=q_mask.shape[0],
                                  dtype=q_mask.dtype)
        input_layers.append(mask)
        input_tensors.append(q_mask)

      transformer_kwargs = {
          'output_layer_size': output_layer_size,
          'max_batch_size': max_batch_size,
          'use_scale': use_scale,
          'num_attention_heads': attention_heads,
          'attention_head_size': attention_head_size,
          'sequences_per_iter': seq_per_iter,
          'embedding_initializer': 'ones',
          'output_initializer': 'ones'
      }

      # Build an Effective Transformer.
      transformer = EffectiveTransformer(**transformer_kwargs)
      x = transformer(input_layers)[0]

      model = keras.Model(inputs=input_layers, outputs=x)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.1)
      model.compile(opt, 'mse')

      history = model.fit(input_tensors,
                          np.ones((from_sequences.shape[0], output_layer_size),
                                  dtype=dtype),
                          batch_size=from_sequences.shape[0],
                          epochs=3,
                          verbose=False)

      # Check losses are decreasing monotonically.
      losses = history.history['loss']
      last_loss = float('inf')
      for l in losses:
        self.assertLess(l, last_loss)
        last_loss = l

  @parameterized.named_parameters(*TEST_CASES)
  def testSerialize(
      self,
      output_layer_size,
      max_batch_size,
      from_seq_lens,  # pylint: disable=unused-argument
      to_seq_lens,  # pylint: disable=unused-argument
      from_row_len,  # pylint: disable=unused-argument
      to_row_len,  # pylint: disable=unused-argument
      dtype,  # pylint: disable=unused-argument
      seq_per_iter,
      use_scale,
      q_mask,  # pylint: disable=unused-argument
      attention_heads,
      attention_head_size):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      transformer_kwargs = {
          'output_layer_size': output_layer_size,
          'max_batch_size': max_batch_size,
          'use_scale': use_scale,
          'num_attention_heads': attention_heads,
          'attention_head_size': attention_head_size,
          'sequences_per_iter': seq_per_iter,
          'embedding_initializer': init_ops.ones_initializer(),
          'output_initializer': init_ops.ones_initializer(),
          'embedding_bias_initializer': init_ops.zeros_initializer(),
          'output_bias_initializer': init_ops.zeros_initializer()
      }

      transformer = EffectiveTransformer(**transformer_kwargs)
      transformer_config = transformer.get_config()

      transformer_2 = EffectiveTransformer.from_config(transformer_config)
      transformer_2_config = transformer_2.get_config()

      self.assertEqual(transformer_config, transformer_2_config)

  @parameterized.named_parameters(*SEQUENCE_PADDING_TEST_CASES)
  def testSequenceUnpad(self, padded_sequence, unpadded_sequence,
                        sequence_lengths):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      num_sequences = len(sequence_lengths)

      @def_function.function(experimental_compile=True)
      def f():
        transformer = EffectiveTransformer(12, 10)
        # pylint: disable=protected-access
        return transformer._unpad(padded_sequence, sequence_lengths,
                                  num_sequences)

      res = strategy.run(f)

    self.assertAllEqual(res, unpadded_sequence)


if __name__ == '__main__':
  googletest.main()

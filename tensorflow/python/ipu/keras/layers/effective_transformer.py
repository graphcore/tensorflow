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
"""
Keras EffectiveTransformer layer
~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu.keras import layers as ipu_layers
from tensorflow.python.ipu.ops.slicing_ops import sequence_slice
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class EffectiveTransformer(Layer):
  """EffectiveTransformer is an implementation of a multihead attention
  network.

  Transformers of this type are described in the following paper:
  https://arxiv.org/abs/1706.03762

  This implementation is optimised for batches of padded sequences, by
  dynamically compressing the input sequences for computationally expensive
  parts of the algorithm. This compression is achieved by the removal of
  padding for those computations that do not rely on a 1:1 relationship
  between the input `to` and `from` sequences.

  For an input sequence tensor `X` of shape `[B, N]`, the algorithm will
  process `X` in compressed chunks of shape `[B', N]`, where `B'` is less than
  or equal to `max_batch_size`. The algorithm output, however, keeps the input
  batch size `B`. Though the maximum batch size of compressed sequences to be
  processed in each chunk is of shape `[B', N]`, the parameter
  `sequences_per_iter` determines the upper limit on the total number of
  compressed sequences to be processed for each `B'` sized batch.

  The distinction between `max_batch_size` and `sequences_per_iter` is of
  importance when a corpus of data has much variance in the length of its
  sequences (the degree of padding in each row). `max_batch_size` determines
  the upper bound on the number of rows of data to be processed in each chunk
  and `sequences_per_iter` determines the upper bound on the number of
  sequences to be compressed into each chunk. This distinction is important
  to consider because a chunk of compressed sequences will need to be
  decompressed at points in the algorithm. This can incur large memory usage
  if the number of compressed sequences to process is high and the uncompressed
  shape unbounded.

  `sequences_per_iter` must be less than or equal to `max_batch_size`.

  Arguments:
    output_layer_size: The number of output units.
    max_batch_size: The upper limit to which additional sequences will
      be compressed into a chunk of data. This is the maximum size of
      the uncompressed sequence tensor.
    use_scale: If True, learn a scale parameter.
    num_attention_heads: The number of attention heads to use for
      multihead attention.
    attention_head_size: The size of each attention head.
    sequences_per_iter: The number of full-sequence equivalents to process
      in each data chunk. Must be less than or equal to `max_batch_size`.
    qkv_activation: The activation function to use for the Query, Key and Value
      embeddings.
    attention_dropout_prob: Dropout probability applied to the attention
      distribution.
    output_activation: The activation function to use for the layer output.
    output_dropout_prob: Dropout probability applied to the layer output.
    layer_norm_output: Whether to apply Layer Normalisation to the output.
  """
  def __init__(self,
               output_layer_size,
               max_batch_size,
               use_scale=False,
               num_attention_heads=1,
               attention_head_size=1,
               sequences_per_iter=1,
               qkv_activation=None,
               attention_dropout_prob=None,
               output_activation=None,
               output_dropout_prob=None,
               layer_norm_output=True,
               **kwargs):
    super().__init__(**kwargs)
    self.max_batch_size = max_batch_size
    self.output_layer_size = output_layer_size
    self.use_scale = use_scale
    self.num_attention_heads = num_attention_heads
    self.attention_head_size = attention_head_size
    self.sequences_per_iter = sequences_per_iter
    self.qkv_activation = qkv_activation
    self.attention_dropout_prob = attention_dropout_prob
    self.output_activation = output_activation
    self.output_dropout_prob = output_dropout_prob
    self.layer_norm_output = layer_norm_output

    self.built = False

    if self.sequences_per_iter > self.max_batch_size:
      raise ValueError(
          "max_batch_size must be greater than sequences_per_iter")

  # pylint: disable=arguments-differ
  def build(self, input_shapes):
    if len(input_shapes) not in (4, 5):
      raise ValueError("EffectiveTransformer must be built with either "
                       "four or five input shapes.")

    qkv_dense_size = self.num_attention_heads * self.attention_head_size

    # Query layer.
    self._q_layer = layers.Dense(
        qkv_dense_size,
        activation=self.qkv_activation,
        kernel_initializer=init_ops.ones_initializer(),
        name='query_layer')

    # Key layer.
    self._k_layer = layers.Dense(
        qkv_dense_size,
        activation=self.qkv_activation,
        kernel_initializer=init_ops.ones_initializer(),
        name='key_layer')

    # Value layer.
    self._v_layer = layers.Dense(
        qkv_dense_size,
        activation=self.qkv_activation,
        kernel_initializer=init_ops.ones_initializer(),
        name='value_layer')

    # Add scale weights.
    if self.use_scale:
      self._scale = self.add_weight(name='scale',
                                    shape=(1),
                                    initializer=init_ops.ones_initializer(),
                                    dtype=self.dtype,
                                    trainable=True)

    # Optionally, dropout can be applied to the attention distribution.
    if not self.attention_dropout_prob is None:
      self._attention_dropout = ipu_layers.Dropout(self.attention_dropout_prob)

    # Output layer.
    self._output_layer = layers.Dense(
        self.output_layer_size,
        activation=self.output_activation,
        kernel_initializer=init_ops.ones_initializer(),
        name='output_layer')

    # Optionally, dropout can be applied to the output layer.
    if not self.output_dropout_prob is None:
      self._output_dropout = ipu_layers.Dropout(self.output_dropout_prob)

    # Optionally, apply layer norm to the output.
    if self.layer_norm_output:
      self._output_layer_norm = ipu_layers.LayerNormalization()
      self._output_layer_norm.build(
          [self.max_batch_size, self.output_layer_size])

    from_dim = input_shapes[0][1]
    to_dim = input_shapes[2][1]

    # Buffer for padded sequences.
    self._from_sequence_buffer = array_ops.zeros(
        (self.max_batch_size, from_dim))

    self._to_sequence_buffer = array_ops.zeros((self.max_batch_size, to_dim))

    # Buffer for unpadded sequences.
    self._from_sequence_buffer_unpadded = array_ops.zeros(
        (self.sequences_per_iter, from_dim))

    self._to_sequence_buffer_unpadded = array_ops.zeros(
        (self.sequences_per_iter, to_dim))

    # Buffer for sequence lengths.
    self._sequence_length_buffer = array_ops.zeros((self.max_batch_size, 1),
                                                   dtype=dtypes.int32)

    # Buffers for padded q, k and v.
    self._padded_q_buffer = array_ops.zeros(
        (self.max_batch_size * from_dim * self.num_attention_heads *
         self.attention_head_size, 1),
        dtype=self._q_layer.dtype)

    self._padded_k_buffer = array_ops.zeros(
        (self.max_batch_size * to_dim * self.num_attention_heads *
         self.attention_head_size, 1),
        dtype=self._k_layer.dtype)

    self._padded_v_buffer = array_ops.zeros(
        (self.max_batch_size * to_dim * self.num_attention_heads *
         self.attention_head_size, 1),
        dtype=self._v_layer.dtype)

    super().build(input_shapes)

  # pylint: disable=arguments-differ
  def call(self, inputs, training=True):
    if len(inputs) not in (4, 5):
      raise ValueError(
          "EffectiveTransformer must take either four or five inputs.")

    from_sequences = ops.convert_to_tensor(inputs[0])
    to_sequences = ops.convert_to_tensor(inputs[2])
    from_sequence_lengths = ops.convert_to_tensor(inputs[1])
    to_sequence_lengths = ops.convert_to_tensor(inputs[3])

    # Check that the from/to tensors are compatible.
    if len(from_sequences.shape) != len(to_sequences.shape):
      raise ValueError("from_sequences and to_sequences must be equal rank")

    if from_sequences.shape[0] != to_sequences.shape[0]:
      raise ValueError("from_sequences and to_sequences must contain an equal "
                       "number of sequences")

    if not (from_sequence_lengths.shape[0] == from_sequences.shape[0]
            and from_sequence_lengths.shape == to_sequence_lengths.shape):
      raise ValueError(
          "from_sequence_lengths and to_sequence_lengths must "
          "have lengths equal to the number of sequences provided.")

    if from_sequences.dtype != to_sequences.dtype:
      raise ValueError("from_sequences and to_sequences must be equal types")

    # If no per-query attention head mask has been provided, just generate
    # all ones. If provided, check it is the correct shape.
    mask_shape = (from_sequences.shape[0], self.num_attention_heads)
    q_mask = inputs[4] if len(inputs) == 5 else None
    if q_mask is None:
      q_mask = array_ops.constant(True, shape=mask_shape, dtype=dtypes.bool)
    q_mask = ops.convert_to_tensor(q_mask)

    if q_mask.shape != mask_shape:
      raise ValueError("q_mask must have shape "
                       "[num_sequences, num_attention_heads]")

    # Context layer output.
    # [num_sequences, from_dim, num_heads, head_size]
    context = array_ops.zeros(
        (from_sequences.shape[0], from_sequences.shape[1],
         self.num_attention_heads, self.attention_head_size),
        dtype=dtypes.float32)

    context = control_flow_ops.while_loop(
        self._main_loop_condition,
        self._main_loop_body,
        loop_vars=[
            context, from_sequences, from_sequence_lengths, to_sequences,
            to_sequence_lengths, q_mask, 0
        ],
        maximum_iterations=self.max_batch_size)[0]

    context_flat = array_ops.reshape(
        context,
        (from_sequences.shape[0],
         from_sequences.shape[1] * self.num_attention_heads * \
        self.attention_head_size))

    output = self._output_layer(context_flat)
    if hasattr(self, '_output_dropout'):
      output = self._output_dropout(output)[0]

    if hasattr(self, '_output_layer_norm'):
      output = self._output_layer_norm(output, training=training)

    return output, context

  def get_config(self):
    config = {
        'output_layer_size': self.output_layer_size,
        'max_batch_size': self.max_batch_size,
        'use_scale': self.use_scale,
        'num_attention_heads': self.num_attention_heads,
        'attention_head_size': self.attention_head_size,
        'sequences_per_iter': self.sequences_per_iter,
        'qkv_activation': self.qkv_activation,
        'attention_dropout_prob': self.attention_dropout_prob,
        'output_activation': self.output_activation,
        'output_dropout_prob': self.output_dropout_prob
    }

    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _main_loop_condition(
      self,
      context,  # pylint: disable=unused-argument
      from_sequences,
      from_sequence_lengths,  # pylint: disable=unused-argument
      to_sequences,  # pylint: disable=unused-argument
      to_sequence_lengths,  # pylint: disable=unused-argument
      q_mask,  # pylint: disable=unused-argument
      num_processed):
    return math_ops.less(num_processed, from_sequences.shape[0])

  def _main_loop_body(self, context, from_sequences, from_sequence_lengths,
                      to_sequences, to_sequence_lengths, q_mask,
                      num_processed):
    #num_processed = array_ops.stop_gradient(num_processed)

    # Get the slicing indices for this iteration.
    min_idx, max_idx = self._get_compressed_sequence_bounds(
        from_sequence_lengths, to_sequence_lengths, num_processed,
        from_sequences.shape, to_sequences.shape)

    num_processed_iter = max_idx - min_idx

    # Slice out some sequences.
    from_seq_batch = self._slice_and_unpad(from_sequences,
                                           from_sequence_lengths, min_idx,
                                           max_idx)

    to_seq_batch = self._slice_and_unpad(to_sequences, to_sequence_lengths,
                                         min_idx, max_idx)

    # Compute q, k and v embeddings.
    q = self._compute_embedding(from_seq_batch, self._q_layer)
    k = self._compute_embedding(to_seq_batch, self._k_layer)
    v = self._compute_embedding(to_seq_batch, self._v_layer)

    # Repad [batch_size, sequence_length, num_attn_heads, attn_head_size]
    q = self._pad_embedding(q, from_sequence_lengths, min_idx, max_idx, 'q')
    k = self._pad_embedding(k, to_sequence_lengths, min_idx, max_idx, 'k')
    v = self._pad_embedding(v, to_sequence_lengths, min_idx, max_idx, 'v')

    # Compute attention probabilities for the sliced queries.
    scale = 1
    if self.use_scale:
      scale = self._scale

    batch_attention_probabilities = \
      self._calculate_attention_probabilities(q, k, scale)

    # Apply the attention probabilities - context layer.
    batch_context = self._apply_attention_probabilities(
        batch_attention_probabilities, v)

    # Apply q_mask if provided.
    batch_context = self._apply_q_mask(batch_context, q_mask, min_idx, max_idx)

    # If a dropout probability was specified.
    if hasattr(self, '_attention_dropout'):
      batch_context = self._attention_dropout(batch_context)[0]

    batch_context = array_ops.transpose(batch_context, [0, 2, 1, 3])
    context = sequence_slice(context, batch_context, [num_processed_iter], [0],
                             [num_processed], False)

    return context, from_sequences, from_sequence_lengths, \
      to_sequences, to_sequence_lengths, q_mask, \
      num_processed + num_processed_iter

  def _get_sequence_buffer(self, sequences, padded):
    d = sequences.shape[1]

    if d == self._from_sequence_buffer.shape[1] and padded:
      if self._from_sequence_buffer.dtype != sequences.dtype:
        with ops.init_scope():
          self._from_sequence_buffer = array_ops.zeros_like(
              self._from_sequence_buffer, dtype=sequences.dtype)
      return self._from_sequence_buffer

    if d == self._to_sequence_buffer.shape[1] and padded:
      if self._to_sequence_buffer.dtype != sequences.dtype:
        with ops.init_scope():
          self._to_sequence_buffer = array_ops.zeros_like(
              self._to_sequence_buffer, dtype=sequences.dtype)
      return self._to_sequence_buffer

    if d == self._from_sequence_buffer_unpadded.shape[1] and not padded:
      if self._from_sequence_buffer_unpadded.dtype != sequences.dtype:
        with ops.init_scope():
          self._from_sequence_buffer_unpadded = array_ops.zeros_like(
              self._from_sequence_buffer_unpadded, dtype=sequences.dtype)
      return self._from_sequence_buffer_unpadded

    if d == self._to_sequence_buffer_unpadded.shape[1] and not padded:
      if self._to_sequence_buffer_unpadded.dtype != sequences.dtype:
        with ops.init_scope():
          self._to_sequence_buffer_unpadded = array_ops.zeros_like(
              self._to_sequence_buffer_unpadded, dtype=sequences.dtype)
      return self._to_sequence_buffer_unpadded

    raise RuntimeError("Internal sequence buffer error.")

  def _get_compressed_sequence_bounds(self, from_seq_lengths, to_seq_lengths,
                                      idx, from_dim, to_dim):
    """
    Given a number of sequences to process in an iteration (number of rows
    in the sequence tensor, rather than real sequences), we want to pack as
    many real sequences into the batch as we can. So, we need to remove
    padding and shift elements accordingly (see `_unpad`).

    As the `from` and `to` tensors must have an equal number of sequences, so
    must their packed batches (as returned by this function). So the output of
    this function is a pair of indices between which both can be sliced to
    maximise the number of real sequences whilst maintaining the 1:1
    relationship between them (`from` and `to`).
    """
    # Compute the maximum number of elements for both the from and to sequence
    # batches.

    from_max = self.sequences_per_iter * from_dim[1]
    to_max = self.sequences_per_iter * to_dim[1]

    from_len = array_ops.squeeze(from_seq_lengths)
    to_len = array_ops.squeeze(to_seq_lengths)

    def _loop_condition(end_idx, num_used_from, num_used_to):
      def _check_bounds():
        # Will the next step take us over either limit?
        will_not_exceed_from = \
          math_ops.less(num_used_from + from_len[end_idx], from_max)

        will_not_exceed_to = \
          math_ops.less(num_used_to + to_len[end_idx], to_max)

        a = math_ops.logical_and(will_not_exceed_from, will_not_exceed_to)
        b = math_ops.less(end_idx - idx, self.max_batch_size)
        return math_ops.logical_and(a, b)

      return control_flow_ops.cond(math_ops.less(end_idx, from_len.shape[0]),
                                   lambda: _check_bounds(), lambda: False)  # pylint: disable=unnecessary-lambda

    def _loop_body(end_idx, num_used_from, num_used_to):
      num_used_from += from_len[end_idx]
      num_used_to += to_len[end_idx]
      end_idx += 1

      return end_idx, num_used_from, num_used_to

    res = control_flow_ops.while_loop(_loop_condition,
                                      _loop_body,
                                      loop_vars=[idx, 0, 0],
                                      maximum_iterations=self.max_batch_size)

    return idx, res[0] - 1

  def _slice_and_unpad(self, sequences, sequence_lengths, start_idx, end_idx):
    # Slice out the padded sequences.
    num_seq = end_idx - start_idx

    buffer = self._get_sequence_buffer(sequences, padded=True)
    buffer = sequence_slice(buffer, sequences, [num_seq], [start_idx], [0],
                            False)

    # Remove the padding.
    seq = sequence_slice(self._sequence_length_buffer, sequence_lengths,
                         [num_seq], [start_idx], [0], True)
    unpadded = self._unpad(buffer, array_ops.squeeze(seq), self.max_batch_size)

    # Slice into output tensor.
    unpadded_buffer = self._get_sequence_buffer(sequences, padded=False)
    return sequence_slice(unpadded_buffer, unpadded, [self.sequences_per_iter],
                          [0], [0], False)

  def _slice_out_sequences(self, sequences, min_idx, max_idx, buffer_size):
    # To slice out a boolean mask for a sequence, it must first
    # be cast to another int or float type.
    # SequenceSlice only supports [float16, float32 and int32].
    dtype = sequences.dtype
    if dtype == dtypes.bool:
      dtype = dtypes.float16

    n = max_idx - min_idx
    seq_batch = array_ops.zeros((buffer_size, *sequences.shape[1:]),
                                dtype=dtype)
    seq_batch = sequence_slice(seq_batch, math_ops.cast(sequences, dtype), [n],
                               [min_idx], [0], False)

    if seq_batch.dtype != sequences.dtype:
      seq_batch = math_ops.cast(seq_batch, sequences.dtype)

    return seq_batch

  def _compute_embedding(self, sequences, embedding_layer):
    """
    Computes the Q, K or V embedding (embedding_layer) of the input sequences.

    1) Flatten sequences.
    2) Forward pass through embedding layer.
    3) Reshape to:
        [batch_size, sequence_length, num_attn_heads, attn_head_size]

        Note that sequence_length is the second dimension of the sequence
        matrix, so may actually contain multiple sequences
        (due to compression).
    """
    m = sequences.shape[0]
    n = sequences.shape[1]
    qkv_shape = (m, n, self.num_attention_heads, self.attention_head_size)

    seq = array_ops.reshape(sequences, (m * n, 1))
    return array_ops.reshape(embedding_layer(seq), qkv_shape)

  def _calculate_attention_probabilities(self, q, k, scale):
    """
    The attention distribution is the normalised dot product between queries
    and keys. Note that the normalisation to make a proper distribution is
    not performed here, but after masking.

    q input dimensions:
        [batch_size, from_seq_length, num_attn_heads, attn_head_size]

    v input_dimensions:
        [batch_size, to_seq_length, num_attn_heads, attn_head_size]

    output dimensions:
        [batch_size, num_attn_heads, from_seq_len, to_seq_len]

    scale is a 1x1
    """
    qt = array_ops.transpose(q, [0, 2, 1, 3])
    kt = array_ops.transpose(k, [0, 2, 1, 3])
    return math_ops.matmul(qt, kt, transpose_b=True) * scale

  def _apply_attention_probabilities(self, batch_attention_probabilities,
                                     value):
    """
    Applies the attention distribution to the value sequences as the
    product between the attention distribution and the value tensor.

    batch_attention_probabilities dims:
        [batch_size, num_attn_heads, from_seq_len, to_seq_len]

    value dims:
        [batch_size, to_seq_len, num_attn_heads, attn_head_size]

    output dims:
        [batch_size, from_seq_len, num_attn_heads, attn_head_size]

    Normalises batch_attention_probabilities into an actual distribution.
    """

    attention_distribution = nn_ops.softmax(batch_attention_probabilities)
    value_t = array_ops.transpose(value, [0, 2, 1, 3])
    return math_ops.matmul(attention_distribution, value_t)

  def _apply_q_mask(self, attention_probabilities, q_mask, min_idx, max_idx):
    """
    Applies the per-query attention head mask to the attention distribution.

    attention_probabilities dims:
        [batch_size, num_attn_heads, from_seq_len, to_seq_len]

    q_mask dims:
        [batch_size, num_attn_heads]

    output dims:
        [batch_size, num_attn_heads, from_seq_len, to_seq_len]
    """
    q_mask_sliced = self._slice_out_sequences(q_mask, min_idx, max_idx,
                                              self.max_batch_size)
    q_mask_sliced = math_ops.cast(q_mask_sliced,
                                  dtype=attention_probabilities.dtype)
    q_mask_sliced = array_ops.expand_dims(q_mask_sliced, 2)
    q_mask_sliced = array_ops.expand_dims(q_mask_sliced, 3)
    return attention_probabilities * q_mask_sliced

  def _unpad(self, sequences, sequence_lengths, slices_per_iter):
    """
    Given a tensor of sequences of the form

    [[1, 1, 0, 0],
     [2, 0, 0, 0],
     [3, 3, 3, 0],
     [4, 4, 0, 0]]

     and their lengths [2, 1, 3, 2]

     the padding is removed, such that the output is of the form

     [[1, 1, 2, 3],
      [3, 3, 4, 4],
      [0, 0, 0, 0],
      [0, 0, 0, 0]]

    """
    def _slice_back(seq_out, row, row_idx):
      # Insert row into seq_out at row_idx.
      return sequence_slice(seq_out, row, [1], [0], [row_idx], False)

    def _get_row_as_col(seq, idx):
      # Pull a row out of seq at idx, add a leading dimension and transpose
      # it such that it becomes [D, 1], where D is the column dimension of
      # the sequence matrix.
      # We do this as we wish to slice a subset of it's elements and
      # sequence_slice slices along the major dimension.
      row = array_ops.zeros((1, seq.shape[1]), dtype=seq.dtype)
      row = sequence_slice(row, seq, [1], [idx], [0], False)
      return array_ops.transpose(row)

    def _insert_full(seq_out, seq, seq_len, num_written, out_row_idx,
                     slice_no):
      # Get the true sequence length (not the sequence matrix column count).
      n = seq_len[slice_no]

      # Pull out dst and src as [D, 1] tensors.
      dst_row = _get_row_as_col(seq_out, out_row_idx)
      src_row = _get_row_as_col(seq, slice_no)

      # Slice the sequence out of src at position 0 and into dst at the
      # next available position - num_written.
      dst_row = sequence_slice(dst_row, src_row, [n], [0], [num_written],
                               False)

      # Replace the whole row with the updated dst in seq_out.
      seq_out = _slice_back(seq_out, array_ops.transpose(dst_row), out_row_idx)

      # Update the element counter.
      num_written += n

      # If we have filled a row, increment the output index and reset
      # the element counter.
      out_row_idx, num_written = control_flow_ops.cond(
          math_ops.equal(num_written, seq.shape[1]), lambda:
          (out_row_idx + 1, 0), lambda: (out_row_idx, num_written))

      return seq_out, seq, seq_len, num_written, out_row_idx, slice_no

    def _insert_partial(seq_out, seq, seq_len, num_written, out_row_idx,
                        slice_no):
      # Get the true sequence length (not the sequence matrix column count).
      n = seq_len[slice_no]

      # If the current row is full, increment the output index and reset
      # the element counter.
      out_row_idx, num_written = control_flow_ops.cond(
          math_ops.equal(num_written, seq.shape[1]), lambda:
          (out_row_idx + 1, 0), lambda: (out_row_idx, num_written))

      # How many elements of the sequence can we fit on the current row.
      num_partial = seq.shape[1] - num_written

      # Pull out dst and src as [D, 1] tensors.
      dst_row = _get_row_as_col(seq_out, out_row_idx)
      src_row = _get_row_as_col(seq, slice_no)

      # Slice the sequence out of src at position 0 and into dst at the
      # next available position - num_written.
      dst_row = sequence_slice(dst_row, src_row, [num_partial], [0],
                               [num_written], False)

      # Replace the whole row with the updated dst in seq_out.
      seq_out = _slice_back(seq_out, array_ops.transpose(dst_row), out_row_idx)

      # Update the row's capacity, it'll now be zero, so insert the
      # remainder of the sequence on the next row.
      # As we are in this branch, we know that we could only fit a partial
      # sequence on the previous row.
      num_written = 0
      out_row_idx += 1

      # How many elements left to copy?
      num_partial_remaining = n - num_partial

      # As above.
      dst_row = _get_row_as_col(seq_out, out_row_idx)
      dst_row = sequence_slice(dst_row, src_row, [num_partial_remaining],
                               [num_partial], [0], False)
      seq_out = _slice_back(seq_out, array_ops.transpose(dst_row), out_row_idx)

      num_written += num_partial_remaining

      return seq_out, seq, seq_len, num_written, out_row_idx, slice_no

    def _do_slice(seq_out, seq, seq_len, num_written, out_row_idx, slice_no):
      # If the current sequence can be fit into the current row, then enter
      # _insert_full, else _insert_partial.
      return control_flow_ops.cond(
          math_ops.less_equal(seq_len[slice_no], seq.shape[1] - num_written),
          lambda: _insert_full(seq_out, seq, seq_len, num_written, out_row_idx,
                               slice_no),
          lambda: _insert_partial(seq_out, seq, seq_len, num_written,
                                  out_row_idx, slice_no))

    def _loop_body(seq_out, seq, seq_len, num_written, out_row_idx, slice_no):
      seq_out, seq, seq_len, num_written, out_row_idx, slice_no = \
      control_flow_ops.cond(
          math_ops.greater(seq_len[slice_no], 0),
          lambda: _do_slice(seq_out, seq, seq_len, num_written,
                            out_row_idx, slice_no),
          lambda: (seq_out, seq, seq_len, num_written, out_row_idx, slice_no))

      return seq_out, seq, seq_len, num_written, out_row_idx, slice_no + 1

    out_seq = array_ops.zeros_like(sequences)

    num_written = array_ops.zeros((), dtype=dtypes.int32)
    out_row_idx = array_ops.zeros((), dtype=dtypes.int32)
    slice_no = array_ops.zeros((), dtype=dtypes.int32)
    res = control_flow_ops.while_loop(lambda *_: True,
                                      _loop_body,
                                      loop_vars=[
                                          out_seq, sequences, sequence_lengths,
                                          num_written, out_row_idx, slice_no
                                      ],
                                      maximum_iterations=slices_per_iter)

    return res[0]

  def _pad_embedding(self, embedding, sequence_lengths, min_idx, max_idx,
                     embedding_type):
    if embedding_type == 'q':
      dst = self._padded_q_buffer
    elif embedding_type == 'k':
      dst = self._padded_k_buffer
    elif embedding_type == 'v':
      dst = self._padded_v_buffer
    else:
      raise ValueError("Invalid embedding type. Should be q, k or v.")

    n = max_idx - min_idx
    shape = embedding.shape

    flat_len = shape[1] * shape[2] * shape[3]
    src = array_ops.reshape(embedding, (shape[0], flat_len))

    lengths = self._sequence_length_buffer
    lengths = sequence_slice(lengths, sequence_lengths, [n], [min_idx], [0],
                             True)
    lengths *= shape[2] * shape[3]
    lengths_squeezed = array_ops.squeeze(lengths)

    src_offsets = math_ops.cumsum(lengths_squeezed, exclusive=True)
    dst_offsets = math_ops.range(self.max_batch_size) * flat_len

    src = array_ops.reshape(src, (self.sequences_per_iter * flat_len, 1))
    dst = sequence_slice(dst, src, lengths_squeezed, src_offsets, dst_offsets,
                         True)

    return array_ops.reshape(
        dst, (self.max_batch_size, shape[1], shape[2], shape[3]))

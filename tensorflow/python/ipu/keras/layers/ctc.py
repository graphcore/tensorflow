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

from tensorflow.python.keras import layers
from tensorflow.python.ipu.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes


class CTCInferenceLayer(layers.Layer):
  """Computes CTC (Connectionist Temporal Classification) predictions using
  a beam search.
  This implementation is designed and optimized for the IPU and cannot be used
  with other systems.

  Args:
    blank_index: The class index to use for the blank label.
    beam_width: The beam width to use in the beam search.
    top_paths: The number of paths to return.
    from_logits: Whether to expect the input data in the form of logits
        (`True`) or log probabilities (`False`).
        Default value is `False`.
    name: A name for this op. Defaults to "ctc_beam_search" or
        "ctc_beam_search_with_logits".
  """
  def __init__(self,
               blank_index=0,
               beam_width=100,
               top_paths=1,
               from_logits=False,
               name=None):
    super().__init__(name)
    self.blank_index = blank_index
    self.beam_width = beam_width
    self.top_paths = top_paths
    self.infer_function = nn_ops.ctc_beam_search_decoder if from_logits \
                         else nn_ops.ctc_beam_search_decoder_with_log_probs

  def call(self, data, data_length, **kwargs):  # pylint: disable=W0221
    """
    Args:
      data: The data input [max_time, batch_size, num_classes] tensor.
      data_length: A tensor of shape [batch_size] containing the number of
          timesteps in each `data` batch entry.

    Returns:
      A tuple of values:

      * Label probabilities: Negative log probabilities that each path is
        correct.
      * Label lengths: Length of each path of predictions.
      * Decoded labels: The predictions made by the beam search.
    """
    return self.infer_function(data,
                               data_length,
                               beam_width=self.beam_width,
                               top_paths=self.top_paths,
                               blank_index=self.blank_index)


class CTCPredictionsLayer(layers.Layer):
  """
  Computes CTC (Connectionist Temporal Classification) most probable
  predictions.

  Returns the most probable predictions from the
  ctc decoder. This selects the most probable of all predictions returned.
  It also fills the values off the end with the blank index

  This layer does a lot of post processing steps to create the predictions.
  If your model is close to its memory limit it may be worth using the
  CTCInference layer and streaming the results of that off the device and
  performing the processing on the CPU. However this
  will create a larger stream copy that may also cost memory.

  Args:
    blank_index: The class index to use for the blank label.
    beam_width: The beam width to use in the beam search.
    top_paths: The number of paths to return.
    from_logits: Whether to expect the input data in the form of logits
        (`True`) or log probabilities (`False`).
        Default value is `False`.
    name: A name for this op. Defaults to "ctc_beam_search" or
        "ctc_beam_search_with_logits".

  """
  def __init__(self,
               blank_index=0,
               beam_width=100,
               top_paths=1,
               from_logits=False,
               name=None):
    super().__init__(name)
    self._inference_layer = CTCInferenceLayer(blank_index=blank_index,
                                              beam_width=beam_width,
                                              top_paths=top_paths,
                                              from_logits=from_logits,
                                              name=name)

  @staticmethod
  def _select_most_likely_path(probs, predicted, top_paths, lengths):
    # The "probs" tensor actually stores the negative log probability of
    # each path. Hence want smallest per batch to find the
    # most probable
    predicted_shape = array_ops.shape(predicted)
    batch_size = predicted_shape[0]
    max_time = predicted_shape[2]
    if top_paths == 1:
      if lengths is None:
        return array_ops.reshape(predicted, [batch_size, max_time])
      return array_ops.reshape(predicted, [batch_size, max_time]), lengths

    indices = math_ops.argmin(probs, axis=1, output_type=dtypes.int32)
    indices = array_ops.reshape(indices, [batch_size, 1])
    batch_range = array_ops.reshape(
        math_ops.range(batch_size, dtype=dtypes.int32), [batch_size, 1])
    indices = array_ops.concat([batch_range, indices], 1)

    # Feel like there must be a better way to slice the batch without having
    # to create a constant tensor whose values are just the batch dimension
    # in order, but my gather_nd skills aren't good enough to find it for now

    # Alternatively instead of using dynamic slice, we could try do a sort
    # of reduce max/conditional copy, have more copies but not dynamic so
    # might be more efficient
    if lengths is None:
      return array_ops.gather_nd(predicted, indices)
    return array_ops.gather_nd(predicted,
                               indices), array_ops.gather_nd(lengths, indices)

  @staticmethod
  def _mask_out_junk_values(blank_index, best_predictions, lengths):
    # As we can't have dynamically sized tensors set every value in prediction
    # after the length to the blank index.
    predicted_shape = best_predictions.get_shape().as_list()
    batch_size, max_time = predicted_shape[0], predicted_shape[1]

    mask = math_ops.range(max_time, dtype=lengths.dtype)
    mask = array_ops.broadcast_to(mask, [batch_size, max_time])

    lengths = array_ops.reshape(lengths, [batch_size, 1])
    lengths = array_ops.broadcast_to(lengths, [batch_size, max_time])

    mask = math_ops.less(mask, lengths)

    blank_tensor = array_ops.constant(blank_index,
                                      shape=[batch_size, max_time],
                                      dtype=best_predictions.dtype)

    return array_ops.where(mask, best_predictions, blank_tensor)

  def _perform_inference(self, data, data_length, **kwargs):
    probs, lengths, predicted = self._inference_layer.call(
        data, data_length, **kwargs)
    best_predictions, lengths = self._select_most_likely_path(
        probs, predicted, self._inference_layer.top_paths, lengths)
    return self._mask_out_junk_values(self._inference_layer.blank_index,
                                      best_predictions, lengths)

  def call(self, data, data_length, **kwargs):  # pylint: disable=W0221
    """
    Args:
      data: The data input [max_time, batch_size, num_classes] tensor The data
          is expected in the form of log probabilities.
      data_length: A tensor of shape [batch_size] containing the number of
          timesteps in each `data` batch entry. If not provided can only
          perform inference.

    Returns:
      The most probable predictions from the CTC decoder. This selects the most
      probable of all predictions returned. It fills the values off the end with
      the blank index.

    """
    return self._perform_inference(data, data_length, **kwargs)

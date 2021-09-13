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


class CTCLoss(layers.Layer):
  """Computes CTC (Connectionist Temporal Classification) loss.
  This implementation is designed and optimized for the IPU and cannot be used
  with other systems.

  Usage:

  .. code-block:: python

    labels = tf.keras.layers.Input((max_label_length), batch_size=batch_size,
                                   dtype=np.int32, name="labels")
    data = tf.keras.layers.Input((max_time, num_classes),
                                 batch_size=batch_size, dtype=np.float32,
                                 name="data")
    label_length = tf.keras.layers.Input((), batch_size=batch_size,
                                         dtype=np.int32, name="label_length")
    logit_length = tf.keras.layers.Input((), batch_size=batch_size,
                                         dtype=np.int32, name="logit_length")

    dense_layer = tf.keras.layers.Dense(num_classes)
    transpose_layer = tf.keras.layers.Lambda(
        lambda x: keras.backend.permute_dimensions(x, (1, 0, 2)))
    ctc_loss_layer = ipu.keras.losses.CTCLoss(from_logits=True)

    x = dense_layer(data)
    x = transpose_layer(x)
    loss = ctc_loss_layer(labels, x, label_length, logit_length)

    model = ipu.keras.Model((labels, data, label_length, logit_length), loss)
    get_loss_output = lambda y_true, y_pred: y_pred
    model.compile('sgd', loss=get_loss_output)

  Args:
    blank_index: The class index to use for the blank label.
    from_logits: Whether to expect the input data in the form of logits
        (`True`) or log probabilities (`False`).
        Default value is `False`.
  """
  def __init__(self, blank_index=0, from_logits=False, **kwargs):
    super().__init__(**kwargs)
    self.blank_index = blank_index
    self.from_logits = from_logits

  def call(self, labels, data, label_length, data_length, **kwargs):  # pylint: disable=W0221
    """
    Args:
      labels: The labels input [batch_size, max_label_length] tensor.
      data: The data input [max_time, batch_size, num_classes].
      label_length: A tensor of shape [batch_size] containing the number of
          labels in each `labels` batch entry.
      data_length: A tensor of shape [batch_size] containing the number of
          timesteps in each `data` batch entry.
    Returns:
      The calculated loss.
    """
    if self.from_logits:
      loss_function = nn_ops.ctc_loss_v2
    else:
      loss_function = nn_ops.ctc_loss_with_log_probs

    loss = loss_function(labels, data, label_length, data_length,
                         self.blank_index)
    loss = math_ops.reduce_mean(loss)
    loss = array_ops.reshape(loss, [1])
    return loss

  def get_config(self):
    config = {
        'blank_index': self.blank_index,
        'from_logits': self.from_logits,
    }

    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

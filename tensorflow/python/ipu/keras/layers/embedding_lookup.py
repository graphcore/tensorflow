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
"""
Embedding Keras layer
~~~~~~~~~~~~~~~~~~~~~
"""

from functools import reduce
from operator import mul

from tensorflow.python.ipu.keras.layers import ipu_layer
from tensorflow.python.ipu.ops import embedding_ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class Embedding(ipu_layer.IPULayer):
  """
  This is designed to be a replacement for the typical use cases of the
  Keras Embedding layer.

  Args:
    input_dim: int > 0. Size of the vocabulary,
      i.e. maximum integer index + 1.
    output_dim: int >= 0. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings` matrix.

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.

  """

  # pylint: disable=useless-super-delegation
  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):

    kwargs['autocast'] = False
    super(Embedding, self).__init__(**kwargs)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)

    self._check_unsupported(embeddings_regularizer, 'embeddings_regularizer')
    self._check_unsupported(activity_regularizer, 'activity_regularizer')
    self._check_unsupported(embeddings_constraint, 'embeddings_constraint')
    self._check_unsupported(mask_zero, 'mask_zero')
    self._check_unsupported(input_length, 'input_length')

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if len(input_shape) != 2:
      raise ValueError(
          "The input shape should be a tensor of shape [batch, input_length]")

    self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer=self.embeddings_initializer,
                                      name='embeddings')
    self.built = True

  # pylint: disable=arguments-differ
  def call(self, inputs, training=None):
    """
    Perform an embedding lookup.

    Args:
        inputs: An integer tensor of indices into the embedding variable.

    Returns:
        The entries of the embedding tensor corresponding to the ids tensor
        indices.
    """
    del training
    return embedding_ops.embedding_lookup(self.embeddings,
                                          ids=inputs,
                                          name=self.name)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def get_config(self):
    config = {
        'input_dim':
        self.input_dim,
        'output_dim':
        self.output_dim,
        'embeddings_initializer':
        initializers.serialize(self.embeddings_initializer)
    }
    base_config = super(Embedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

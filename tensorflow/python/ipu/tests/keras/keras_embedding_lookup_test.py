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
"""Tests for IPU Embedding layer."""


import numpy as np

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

dataType = np.float32


def _embeddingLookup(instance, params_val, ids_val):
  with ops.device('cpu'):
    params = array_ops.placeholder(dataType, params_val.shape)
    ids = array_ops.placeholder(np.int32, ids_val.shape)
    output = nn.embedding_lookup(params, ids)

  with instance.test_session() as sess:
    return sess.run(output, {params: params_val, ids: ids_val})


def kerasIPUEmbeddingLookup(instance, params_val, ids_val):
  input_dim = params_val.shape[0]
  output_dim = params_val.shape[1]
  with ops.device('/device:IPU:0'):
    layer = ipu.layers.Embedding(
        input_dim,
        output_dim,
        embeddings_initializer=keras.initializers.constant(params_val))
    layer.build(ids_val.shape)

    ids = array_ops.placeholder(np.int32, ids_val.shape)
    output = layer(ids)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    return sess.run(output, {ids: ids_val})


class IPUEmbeddingLookupTest(test.TestCase):
  @test_util.deprecated_graph_mode_only
  def testEmbeddingLookup(self):
    ids = np.array([[1, 2, 3]])
    paras = np.array([[10], [20], [80], [40]])
    emb_lookup_tf = _embeddingLookup(self, paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(self, paras, ids)
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  def testEmbeddingLookupBatchSize2(self):
    ids = np.array([[1, 2, 3], [3, 4, 5]])
    paras = np.array([[10], [20], [80], [40], [50], [60]])
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(self, paras, ids)
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  # Based on ipu/tests/embedding_lookup_test.py
  def testEmbeddingLookupBigGather(self):
    ids = np.arange(0, 8, dtype=np.int32).reshape([1, 8])
    paras = np.arange(2400000, dtype=dataType).reshape([12000, 200])
    result_ipu = kerasIPUEmbeddingLookup(self, paras, ids)
    result_np = np.take(paras, ids, axis=0)
    self.assertAllClose(result_ipu, result_np)
    self.assertEqual(result_ipu.shape, (1, 8, 200))

  def testEmbeddingBadInputShape(self):
    ids = np.arange(0, 16, dtype=np.int32)
    paras = np.arange(25600, dtype=dataType).reshape([32, 200, 4])
    with self.assertRaisesRegexp(ValueError, r'The input shape should be a'):
      kerasIPUEmbeddingLookup(self, paras, ids)


if __name__ == '__main__':
  test.main()

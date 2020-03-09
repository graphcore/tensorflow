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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.python import ipu
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops

dataType = np.float32


def _embeddingLookup(instance, params_val, ids_val):
  with ops.device('cpu'):
    params = array_ops.placeholder(dataType, params_val.shape)
    ids = array_ops.placeholder(np.int32, ids_val.shape)
    output = nn.embedding_lookup(params, ids)

  with instance.test_session() as sess:
    return sess.run(output, {params: params_val, ids: ids_val})


def kerasIPUEmbeddingLookup(instance, params_val, ids_val):
  with ops.device('/device:IPU:0'):
    params = array_ops.placeholder(dataType, params_val.shape)
    ids = array_ops.placeholder(np.int32, ids_val.shape)
    output = ipu.layers.Embedding()(params, ids)

  with instance.test_session() as sess:
    return sess.run(output, {params: params_val, ids: ids_val})


class IPUEmbeddingLookupTest(test.TestCase):
  @test_util.deprecated_graph_mode_only
  def testEmbeddingLookup(self):
    params = np.array([10, 20, 80, 40])
    ids = np.array([1, 2, 3])
    emb_lookup_tf = _embeddingLookup(self, params, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(self, params, ids)

    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  @test_util.deprecated_graph_mode_only
  def testEmbeddingLookupGather(self):
    ids = np.arange(0, 8, dtype=np.int32)
    params = np.arange(2400000, dtype=dataType).reshape([12000, 200])
    result_ipu = kerasIPUEmbeddingLookup(self, params, ids)
    result_np = np.take(params, ids, axis=0)
    np.take(params, ids, axis=0)
    self.assertAllClose(result_ipu, result_np)
    self.assertEqual(result_ipu.shape, (8, 200))

  @test_util.deprecated_graph_mode_only
  def testEmbeddingLookupAutoFlatten(self):
    ids = np.array([[[10, 11], [12, 13], [14, 15], [16, 17]],
                    [[20, 21], [22, 23], [24, 25], [26, 27]],
                    [[30, 31], [32, 33], [34, 35], [36, 37]]],
                   dtype=np.int32)
    params = np.random.uniform(0.0, 1.0, size=(100, 16))
    emb_lookup_tf = _embeddingLookup(self, params, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(self, params, ids)
    self.assertEqual(emb_lookup_ipu.shape, (3, 4, 2, 16))
    self.assertAllClose(emb_lookup_ipu, emb_lookup_tf)

  @test_util.deprecated_graph_mode_only
  def testEmbeddingLookup3(self):
    ids = np.array([4, 8, 15, 16, 23, 42, 8, 4, 15, 16], dtype=np.int32)
    params = np.random.uniform(0.0, 1.0, size=(100, 16))
    emb_lookup_tf = _embeddingLookup(self, params, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(self, params, ids)
    self.assertEqual(emb_lookup_ipu.shape, (10, 16))
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  @test_util.deprecated_graph_mode_only
  def testEmbedding4D(self):
    ids = np.arange(0, 16, dtype=np.int32).reshape([8, 2])
    params = np.arange(25600, dtype=dataType).reshape([32, 200, 4])
    result_ipu = kerasIPUEmbeddingLookup(self, params, ids)
    result_np = np.take(params, ids, axis=0)
    self.assertEqual(result_ipu.shape, (8, 2, 200, 4))
    self.assertAllClose(result_ipu, result_np)


if __name__ == '__main__':
  test.main()

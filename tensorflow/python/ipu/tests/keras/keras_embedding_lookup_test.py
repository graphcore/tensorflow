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

from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn

dataType = np.float32


def kerasIPUEmbeddingLookup(params, ids, name=None):
  layer = ipu.layers.Embedding(name=name)
  layer.build(input_shape=None)

  @def_function.function
  def impl(params, ids):
    return layer(inputs=params, ids=ids)

  return impl(params, ids)


class IPUEmbeddingLookupTest(test.TestCase):
  def testEmbeddingLookup(self):
    paras = constant_op.constant([10, 20, 80, 40])
    ids = constant_op.constant([1, 2, 3])
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_0")

    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  # Based on ipu/tests/embedding_lookup_test.py
  def testEmbeddingLookupGather(self):
    ids = np.arange(0, 8, dtype=np.int32)
    paras = np.arange(2400000, dtype=dataType).reshape([12000, 200])
    result_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_1")
    result_np = np.take(paras, ids, axis=0)
    np.take(paras, ids, axis=0)
    self.assertAllClose(result_ipu, result_np)
    self.assertEqual(result_ipu.shape, (8, 200))

  def testEmbeddingLookupAutoFlatten(self):
    ids = np.array([[[10, 11], [12, 13], [14, 15], [16, 17]],
                    [[20, 21], [22, 23], [24, 25], [26, 27]],
                    [[30, 31], [32, 33], [34, 35], [36, 37]]],
                   dtype=np.int32)
    paras = np.random.uniform(0.0, 1.0, size=(100, 16))
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_2")
    self.assertEqual(emb_lookup_ipu.shape, (3, 4, 2, 16))
    self.assertAllClose(emb_lookup_ipu, emb_lookup_tf)

  def testEmbeddingLookup3(self):
    ids = np.array([4, 8, 15, 16, 23, 42, 8, 4, 15, 16], dtype=np.int32)
    paras = np.random.uniform(0.0, 1.0, size=(100, 16))
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_3")
    self.assertEqual(emb_lookup_ipu.shape, (10, 16))
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  def testEmbedding4D(self):
    ids = np.arange(0, 16, dtype=np.int32).reshape([8, 2])
    paras = np.arange(25600, dtype=dataType).reshape([32, 200, 4])
    result_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_4")
    result_np = np.take(paras, ids, axis=0)
    self.assertEqual(result_ipu.shape, (8, 2, 200, 4))
    self.assertAllClose(result_ipu, result_np)


if __name__ == '__main__':
  test.main()

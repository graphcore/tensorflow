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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import test_util
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.utils import SyntheticDataCategory
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class UseSyntheticDataForTest(test_util.TensorFlowTestCase):
  def _test_with_flags(self, flags, expected_results):
    flags = os.environ.get("TF_POPLAR_FLAGS", "") + " " + flags

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": flags}):
      results = [
          utils.use_synthetic_data_for(SyntheticDataCategory.Seed),
          utils.use_synthetic_data_for(SyntheticDataCategory.Infeed),
          utils.use_synthetic_data_for(SyntheticDataCategory.Outfeed),
          utils.use_synthetic_data_for(SyntheticDataCategory.HostEmbedding),
          utils.use_synthetic_data_for(SyntheticDataCategory.Parameters),
      ]

    self.assertAllEqual(
        expected_results, results,
        'Unexpected results using TF_POPLAR_FLAGS="{}"'.format(flags))

  @test_util.deprecated_graph_mode_only
  def testNoFlags(self):
    flags = ""
    expected_results = [False] * 5
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testUseSyntheticData(self):
    flags = "--use_synthetic_data"
    expected_results = [True] * 5
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testSyntheticDataCategorySeed(self):
    flags = "--synthetic_data_categories=seed"
    expected_results = [False] * 5
    expected_results[0] = True
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testSyntheticDataCategoryInfeed(self):
    flags = "--synthetic_data_categories=infeed"
    expected_results = [False] * 5
    expected_results[1] = True
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testSyntheticDataCategoryOutfeed(self):
    flags = "--synthetic_data_categories=outfeed"
    expected_results = [False] * 5
    expected_results[2] = True
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testSyntheticDataCategoryHostEmbedding(self):
    flags = "--synthetic_data_categories=hostembedding"
    expected_results = [False] * 5
    expected_results[3] = True
    self._test_with_flags(flags, expected_results)

  @test_util.deprecated_graph_mode_only
  def testSyntheticDataCategoryParameters(self):
    flags = "--synthetic_data_categories=parameters"
    expected_results = [False] * 5
    expected_results[4] = True
    self._test_with_flags(flags, expected_results)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

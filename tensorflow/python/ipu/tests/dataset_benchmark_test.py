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
# =============================================================================
import json

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python import ipu


class DatasetBenchmarkTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testWithDataset(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    benchmark_op = ipu.dataset_benchmark.dataset_benchmark(dataset, 10, 1000)

    with self.session() as sess:
      j_str = sess.run(benchmark_op)
      j = json.loads(j_str[0])
      self.assertAllEqual(len(j["epochs"]), 10)
      for x in j["epochs"]:
        for field in x:
          self.assertAllGreater(x[field], 0.0)

  @test_util.deprecated_graph_mode_only
  def testWithInfeed(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    benchmark_op = ipu.dataset_benchmark.infeed_benchmark(infeed_queue, 5, 256)

    with self.session() as sess:
      j_str = sess.run(benchmark_op)
      j = json.loads(j_str[0])
      print(j)
      self.assertAllEqual(len(j["epochs"]), 5)
      for x in j["epochs"]:
        for field in x:
          self.assertAllGreater(x[field], 0.0)

  @test_util.deprecated_graph_mode_only
  def testShape0Dim(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 0])
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    benchmark_op = ipu.dataset_benchmark.infeed_benchmark(infeed_queue, 5, 256)

    with self.session() as sess:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          'Detected a tensor in the dataset with a dimension of size 0'):
        sess.run(benchmark_op)


if __name__ == "__main__":
  googletest.main()

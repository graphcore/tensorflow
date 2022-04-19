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
import numpy as np

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.compat.v1 import data as compat_v1_data


class DatasetOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testBufferDataset(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    dataset = dataset.take(3)
    dataset = ipu.data.ops.dataset_ops.BufferDataset(dataset, 2)
    itr = compat_v1_data.make_one_shot_iterator(dataset)

    next_data = itr.get_next()
    with self.session() as sess:
      self.assertAllEqual(sess.run(next_data)[0], np.zeros([4, 4]))
      self.assertAllEqual(sess.run(next_data)[0], np.ones([4, 4]))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(sess.run(next_data))


if __name__ == "__main__":
  googletest.main()

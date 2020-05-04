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

from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu.scopes import ipu_shard
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GclTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testInvaildNumberOfIoTilesRaisesException(self):
    def my_net(a):
      with ipu_shard(0):
        aa = math_ops.matmul(a, a)
      with ipu_shard(1):
        return math_ops.matmul(aa, aa)

    inputs = array_ops.placeholder(np.float32, [2, 2])
    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[inputs])

    with session.Session() as sess:
      ReportJSON(self, sess, sharded=True)
      with test.mock.patch.dict("os.environ", {"GCL_NUM_IO_TILES": "foo"}):
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "Cannot parse value of the environment variable "
            "GCL_NUM_IO_TILES as an integer: foo"):
          sess.run(compiled_net, {inputs: np.ones(inputs.shape)})

    with session.Session() as sess:
      ReportJSON(self, sess, sharded=True)
      with test.mock.patch.dict("os.environ", {"GCL_NUM_IO_TILES": "-1"}):
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "-1 is an invalid number of IO tiles"):
          sess.run(compiled_net, {inputs: np.ones(inputs.shape)})

  @test_util.deprecated_graph_mode_only
  def testIoTilesAreExcludedFromShard(self):
    def my_net(a, b):
      with ipu_shard(0):
        aa = math_ops.matmul(a, a, transpose_b=True, name="aa")
      with ipu_shard(1):
        bb = math_ops.matmul(b, b, transpose_b=True, name="bb")
      return aa, bb

    input_a = array_ops.placeholder(np.float32, [1216, 1])
    input_b = array_ops.placeholder(np.float32, [1216, 1])

    with ops.device("/device:IPU:0"):
      compiled_net = ipu_compiler.compile(my_net, inputs=[input_a, input_b])

    num_io_tiles = 128

    with test.mock.patch.dict("os.environ",
                              {"GCL_NUM_IO_TILES": str(num_io_tiles)}):

      with session.Session() as sess:
        report = ReportJSON(self, sess, sharded=True)
        report.reset()

        sess.run(compiled_net, {
            input_a: np.ones(input_a.shape),
            input_b: np.ones(input_b.shape)
        })

        report.parse_log()
        num_compute_tiles = report.get_num_tiles_per_ipu() - num_io_tiles
        for t in report.get_tensor_map().all_tensors():
          self.assertLessEqual(len(t.tiles), num_compute_tiles)


if __name__ == "__main__":
  test.main()

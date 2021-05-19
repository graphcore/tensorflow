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

from tensorflow.python import ipu
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.ops import summary_ops
from tensorflow import function, constant, float32


class ContribIpuOpsTest(test_util.TensorFlowTestCase):
  @test_util.run_v2_only()
  def testSummary(self):
    # Create ipu config
    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Create dummy graph
      @function()
      def test_fn(a, b):
        return a + b

      # Init test vars
      a = constant([1.0], dtype=float32)
      b = constant([2.0], dtype=float32)

      # Run graph
      result = strategy.experimental_run_v2(test_fn, args=(a, b))

      # Extract summary
      reports = summary_ops.get_ipu_reports()

      # From the above summary extarct the compile report
      compile_report = ipu.utils.extract_compile_reports(reports)[0][1]

      # Check that everything is fine
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(compile_report) > 100)


if __name__ == "__main__":
  test = ContribIpuOpsTest()
  test.testSummary()

#  Copyright 2020 The TensorFlow Authors. All Rights Reserved.

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================

from absl.testing import parameterized
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

disable_v2_behavior()


class ContribIpuGetConfigOpTest(test_util.TensorFlowTestCase,
                                parameterized.TestCase):
  @test_util.deprecated_graph_mode_only
  def testNoConfig(self):
    # Verify that an exception is thrown if called before configuring.
    with self.assertRaisesRegex(RuntimeError, "No IPU devices configured."):
      _ = ipu.utils.get_ipu_config()

  @test_util.deprecated_graph_mode_only
  def testGetConfig(self):
    # Generate a simple IPU config.
    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=True)
    cfg = ipu.utils.auto_select_ipus(cfg, [2, 2])

    # Configure IPU.
    ipu.utils.configure_ipu_system(cfg)

    # Get back serialised IpuOption instances. One instance per
    # stream executor is expected.
    sess = sl.Session()
    result = ipu.utils.get_ipu_config(sess)

    # Each element in the tensor is a serialised IpuOption belonging to
    # a different device/executor.
    self.assertTrue(len(result) == 2)

    for opt in result:
      # Verify that compile_ipu_code is True.
      self.assertTrue(opt.ipu_model_config.compile_ipu_code)

      # Verify that the device has two IPU's attached.
      self.assertTrue(len(opt.device_config) == 2)
      for dev in opt.device_config:
        self.assertTrue(dev.auto_count == 2)

      # Verify that this IpuOption was user created.
      self.assertTrue(opt.creator_id == 1)


if __name__ == "__main__":
  googletest.main()

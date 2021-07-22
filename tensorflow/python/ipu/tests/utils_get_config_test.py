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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = True
    cfg.auto_select_ipus = [2, 4]

    # Configure IPU.
    cfg.configure_ipu_system()

    # Get back serialised IpuOption instances. One instance per
    # stream executor is expected.
    sess = sl.Session()
    result = ipu.utils.get_ipu_config(sess)

    # Each element in the tensor is a serialised IpuOption belonging to
    # a different device/executor.
    self.assertEqual(len(result), 2)

    for opt in result:
      # Verify that compile_ipu_code is True.
      self.assertTrue(opt.ipu_model_config.compile_ipu_code)

      # Verify that the device has two IPU's attached.
      self.assertEqual(len(opt.device_config), 2)
      for i, dev in enumerate(opt.device_config):
        self.assertEqual(dev.auto_count, 2**(i + 1))

      # Verify that this IpuOption was user created.
      self.assertTrue(opt.creator_id == 1)

  @test_util.deprecated_graph_mode_only
  def testGetNumberOfIpus(self):
    # Generate a simple IPU config.
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = True
    cfg.auto_select_ipus = [2, 4]
    cfg.configure_ipu_system()

    self.assertEqual(ipu.utils.get_num_of_ipus_in_device("/device:IPU:0"), 2)
    self.assertEqual(ipu.utils.get_num_of_ipus_in_device("/device:IPU:1"), 4)

  @test_util.deprecated_graph_mode_only
  def testGetNumberOfIpusNoConfig(self):
    self.assertEqual(ipu.utils.get_num_of_ipus_in_device("/device:IPU:0"), 1)


if __name__ == "__main__":
  googletest.main()

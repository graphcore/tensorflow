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
from tensorflow.python.framework import test_util

from tensorflow.python.framework import errors
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python import ipu


class IPUReconfigureTest(test_util.TensorFlowTestCase):
  @classmethod
  def setUpClass(cls):
    cls.first_cfg = ipu.config.IPUConfig()
    cls.first_cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cls.first_cfg.auto_select_ipus = [1, 1]
    cls.first_cfg.ipu_model.compile_ipu_code = True

    cls.second_cfg = ipu.config.IPUConfig()
    cls.second_cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cls.second_cfg.auto_select_ipus = [1, 2, 1]

  def testChangingConfigWithoutResetRaises(self):
    self.first_cfg.configure_ipu_system()

    self.assertRaises(errors.FailedPreconditionError,
                      ipu.config.configure_ipu_system,
                      self.second_cfg,
                      reset_configuration=False)

  def testCanChangeConfigurationAfterReset(self):
    self.first_cfg.configure_ipu_system()
    ipu.config.reset_ipu_configuration()

    self.second_cfg.configure_ipu_system()
    pb = self.second_cfg._create_protobuf()  # pylint: disable=protected-access

    configs = ipu.config.get_ipu_config()
    for config in configs:
      self.assertEqual(config, pb)

  def testNoConfigAfterResetting(self):
    self.first_cfg.configure_ipu_system()
    ipu.config.reset_ipu_configuration()

    # If no devices are configured then get_ipu_config throws.
    self.assertRaises(RuntimeError, ipu.config.get_ipu_config)

  def testCanResetEmptyConfiguration(self):
    try:
      ipu.config.reset_ipu_configuration()
    except Exception:  # pylint: disable=broad-except
      self.fail(
          "Unexpected exception thrown when resetting empty configuration")

  def testResetsAutomaticallyWhenConfiguring(self):
    self.first_cfg.configure_ipu_system()

    self.second_cfg.configure_ipu_system()
    pb = self.second_cfg._create_protobuf()  # pylint: disable=protected-access
    for config in ipu.config.get_ipu_config():
      self.assertEqual(config, pb)

  def testCanResetWithinIPUDeviceContext(self):
    ''' It's not uncommon for ipu.config.configure_ipu_system to be called from
    within a ipu device scope. This test is to make sure that the reset works
    when that happens, since the kernels used for resetting are CPU only'''

    self.first_cfg.configure_ipu_system()

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      try:
        ipu.config.reset_ipu_configuration()
      except Exception:  # pylint: disable=broad-except
        self.fail(
            "Unexpected exception thrown when resetting configuration from "
            "within IPU device context")

  def testOnlyResetOpsAreRun(self):
    ''' ipu.config.configure_ipu_system can be called with a non-empty default
    graph. This test is to make sure that the reset doesn't try to run Ops
    in the existing graph '''

    graph = ops.Graph()
    with graph.as_default():
      # If we try and run this constant it'll fail since this device
      # doesn't exist.
      with ipu.scopes.ipu_scope("/device:IPU:9999999"):
        constant_op.constant(0.5)

      try:
        ipu.config.reset_ipu_configuration()
      except Exception:  # pylint: disable=broad-except
        self.fail(
            "Unexpected exception thrown when resetting configuration with "
            "a non-empty default graph")


if __name__ == "__main__":
  googletest.main()

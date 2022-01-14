# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""A pytest style test checking the format of the error when TF fails
to attach to a device."""
import multiprocessing
import signal
import time

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework.errors_impl import InternalError


def attach(poplar_device, wait_for_signal=True):
  """Attaches to IPUs and optionally waits for a signal"""
  cfg = ipu.config.IPUConfig()
  cfg.select_ipus = [poplar_device]
  cfg.configure_ipu_system()

  # Hold the device until termination. Handle SIGTERM gracefully
  # to avoid weird device behaviour.
  class TerminateCheck:
    def __init__(self):
      self.terminated = False

    def terminate(self, _signo, _stack_frame):
      self.terminated = True

  tc = TerminateCheck()
  signal.signal(signal.SIGTERM, tc.terminate)

  while not tc.terminated and wait_for_signal:
    time.sleep(0.1)


class TestAttachFailure(xla_test.XLATestCase):  # pylint: disable=abstract-method
  @tu.test_uses_ipus(num_ipus=1)
  def testUnavailableDevice(self):
    """Checks that the correct error message is thrown when failing to
    attach to IPUs.

    This test MUST run on it's own to guarantee that the Poplar Device with
    ID 0 is available.
    """
    poplar_device = 0
    # Poplar device 0 has 1 IPU.
    ipu_count = 1
    # Attach to a single IPU on ordinal 0
    proc = multiprocessing.Process(target=attach, args=(poplar_device,))
    proc.start()
    time.sleep(5)  # Wait for the process to have attached to the IPU
    try:
      # Trigger failure by attaching to the same poplar device
      with self.assertRaisesRegex(InternalError, f"expected.*{ipu_count} IPU"):
        attach(poplar_device, wait_for_signal=False)
    finally:
      # Need this to avoid a hang if there is an error.
      proc.terminate()


if __name__ == "__main__":
  googletest.main()

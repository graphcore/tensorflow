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
# =============================================================================

import os
from tensorflow.python.ipu import config


def get_ci_num_ipus():
  return int(os.getenv('TF_IPU_COUNT', "0"))


def has_ci_ipus():
  return get_ci_num_ipus() > 0


def add_hw_ci_connection_options(opts):
  opts.device_connection.enable_remote_buffers = True
  opts.device_connection.type = config.DeviceConnectionType.ON_DEMAND


def test_may_use_ipus_or_model(num_ipus, func=None):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
  """Test decorator for indicating that a test can run on both HW and Poplar
  IPU Model.
  Args:
  * num_ipus: number of IPUs required by the test.
  * func: the test function.
  """
  return test_uses_ipus(num_ipus=num_ipus, allow_ipu_model=True, func=func)


def test_uses_ipus(num_ipus, allow_ipu_model=False, func=None):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
  """Test decorator for indicating how many IPUs the test requires. Allows us
  to skip tests which require too many IPUs.

  Args:
  * num_ipus: number of IPUs required by the test.
  * allow_ipu_model: whether the test supports IPUModel so that it can be
    executed without hardware.
  * func: the test function.
  """
  def decorator(f):
    def decorated(self, *args, **kwargs):
      num_available_ipus = get_ci_num_ipus()
      if num_available_ipus < num_ipus and not allow_ipu_model:
        self.skipTest(f"Requested {num_ipus} IPUs, but only "
                      f"{num_available_ipus} are available.")
      if num_available_ipus >= num_ipus:
        assert not ("use_ipu_model" in os.getenv(
            'TF_POPLAR_FLAGS',
            "")), "Do not set use_ipu_model when running HW tests."
      return f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def skip_on_hw(func):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
  """Test decorator for skipping tests which should not be run on HW."""
  def decorator(f):
    def decorated(self, *args, **kwargs):
      if has_ci_ipus():
        self.skipTest("Skipping test on HW")

      return f(self, *args, **kwargs)

    return decorated

  return decorator(func)


def skip_if_not_enough_ipus(self, num_ipus):
  num_available_ipus = get_ci_num_ipus()
  if num_available_ipus < num_ipus:
    self.skipTest(f"Requested {num_ipus} IPUs, but only "
                  f"{num_available_ipus} are available.")


def skip_with_asan(reason):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
  """Test decorator for skipping tests which should not be run with AddressSanitizer."""
  if not isinstance(reason, str):
    raise TypeError("'reason' should be string, got {}".format(type(reason)))

  def decorator(f):
    def decorated(self, *args, **kwargs):
      if "ASAN_OPTIONS" in os.environ:
        self.skipTest(reason)

      return f(self, *args, **kwargs)

    return decorated

  return decorator

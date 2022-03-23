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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest

# This test rounds a 1000 element float32 by casting to a float16. Stochastic
# rounding will produce one of 2 values for each element.
#
# We test:
# - that 2 devices do not have the same rounding with the initial seed
# - that repeatedly rounding produces different values
# - setting the seed to a specific value generates the same rounding
# - setting the seed to different values generates different rounding
# - setting the seed after generating multiple rounding events restarts the
#   same sequence


class TestSeedControl(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_reset_seed(self):
    # The host side
    inp = array_ops.placeholder(np.float32, [1000])

    # The device side (0)
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      out0 = math_ops.cast(inp, dtype=np.float16)

    # The device side (1)
    with ipu.scopes.ipu_scope('/device:IPU:1'):
      out1 = math_ops.cast(inp, dtype=np.float16)

    # Configure the hardware
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [1, 1]
    tu.add_hw_ci_connection_options(config)
    config.floating_point_behaviour.inv = True
    config.floating_point_behaviour.div0 = True
    config.floating_point_behaviour.oflo = True
    config.floating_point_behaviour.esr = \
      ipu.config.StochasticRoundingBehaviour.ON
    config.floating_point_behaviour.nanoo = True
    config.compilation_poplar_options = {'target.deterministicWorkers': 'true'}
    config.configure_ipu_system()

    with session_lib.Session() as sess:

      in_data = np.array([0.1] * 1000)

      # Each device doesn't generate same initial rounding
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out1, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 doesn't generate the same rounding repeatedly
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 1 doesn't generate the same rounding repeatedly
      res1 = sess.run(out1, {inp: in_data}).astype(np.float32)
      res2 = sess.run(out1, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 can reset seed to generate the same result
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)

      # Device 0 can reset with different seeds
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(2)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertNotAllEqual(res1, res2)

      # Device 0 can do multiple steps and then have the seed reset
      ipu.utils.reset_ipu_seed(1)
      res1 = sess.run(out0, {inp: in_data}).astype(np.float32)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)
      ipu.utils.reset_ipu_seed(1)
      res2 = sess.run(out0, {inp: in_data}).astype(np.float32)
      self.assertAllEqual(res1, res2)


if __name__ == "__main__":
  googletest.main()

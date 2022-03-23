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
import os
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


@test_util.deprecated_graph_mode_only
class TestFloatingPointControlBits(test_util.TensorFlowTestCase):
  @staticmethod
  def _configure(
      invalid_operation=False,
      division_by_zero=False,
      overflow=False,
      stochastic_rounding=ipu.config.StochasticRoundingBehaviour.OFF,
      nan_overflow=False,
      experimental_prng_stability=False,
      ipu_count=1):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = ipu_count
    cfg.floating_point_behaviour.inv = invalid_operation
    cfg.floating_point_behaviour.div0 = division_by_zero
    cfg.floating_point_behaviour.oflo = overflow
    cfg.floating_point_behaviour.esr = stochastic_rounding
    cfg.floating_point_behaviour.nanoo = nan_overflow
    cfg.experimental.enable_prng_stability = experimental_prng_stability
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

  @staticmethod
  def _test_invalid_operation():
    ph = array_ops.placeholder(np.float16, [1])

    def model(x):
      return math_ops.cast(x, dtype=np.int32)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(model, inputs=[ph])

    with sl.Session() as sess:
      return sess.run(r, {ph: [np.nan]})

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_invalid_operation(self):
    self._configure()
    self.assertEqual(self._test_invalid_operation(), [-2147483648])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_invalid_operation_exception(self):
    self._configure(invalid_operation=True)
    with self.assertRaisesRegex(Exception, "Tiles in excepted state"):
      self._test_invalid_operation()

  @staticmethod
  def _test_division_by_zero():
    ph0 = array_ops.placeholder(np.float16, [1])
    ph1 = array_ops.placeholder(np.float16, [1])

    def model(x, y):
      return x / y

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(model, inputs=[ph0, ph1])

    with sl.Session() as sess:
      return sess.run(r, {ph0: [10000.], ph1: [0.]})

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_division_by_zero(self):
    self._configure()
    self.assertTrue(np.isnan(self._test_division_by_zero()))

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_division_by_zero_exception(self):
    self._configure(division_by_zero=True)
    with self.assertRaisesRegex(Exception, "Tiles in excepted state"):
      self._test_division_by_zero()

  @staticmethod
  def _test_overflow():
    ph0 = array_ops.placeholder(np.float16, [1])
    ph1 = array_ops.placeholder(np.float16, [1])

    def model(x, y):
      return x * y

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(model, inputs=[ph0, ph1])

    with sl.Session() as sess:
      return sess.run(r, {ph0: [10000.], ph1: [10000.]})

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_overflow(self):
    self._configure()
    self.assertEqual(self._test_overflow(), np.array([65500.],
                                                     dtype=np.float16))

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_overflow_exception(self):
    self._configure(overflow=True)
    with self.assertRaisesRegex(Exception, "Tiles in excepted state"):
      self._test_overflow()

  @staticmethod
  def _test_stochastic_rounding():
    ph0 = array_ops.placeholder(np.float16, [4096])
    ph1 = array_ops.placeholder(np.float16, [4096])

    def model(x, y):
      return x + y

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(model, inputs=[ph0, ph1])

    with sl.Session() as sess:
      out = sess.run(r, {ph0: np.full([4096], 1), ph1: np.full([4096], 1e-3)})
    return np.histogram(out, bins=2)[0]

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_stochastic_rounding(self):
    self._configure()
    hist = self._test_stochastic_rounding()
    self.assertEqual(hist[0], 0)
    self.assertEqual(hist[1], 4096)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_stochastic_rounding_replica_only(self):
    self._configure(stochastic_rounding=ipu.config.StochasticRoundingBehaviour.
                    REPLICA_IDENTICAL_ONLY,
                    experimental_prng_stability=True,
                    ipu_count=2)
    hist = self._test_stochastic_rounding()
    # Result should be same as  using StochasticRoundingBehaviour.ON since our graph
    # and data are replica identical.
    self.assertTrue(hist[0] > 0)
    self.assertTrue(hist[1] > 0)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_stochastic_rounding_enabled(self):
    self._configure(
        stochastic_rounding=ipu.config.StochasticRoundingBehaviour.ON)
    hist = self._test_stochastic_rounding()
    self.assertTrue(hist[0] > 0)
    self.assertTrue(hist[1] > 0)

  @staticmethod
  def _test_scoped_stochastic_rounding():
    ph0 = array_ops.placeholder(np.float16, [4096])
    ph1 = array_ops.placeholder(np.float16, [4096])

    outputs = {}

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      out = ph0 + ph1
      outputs[out] = True
      with ipu.scopes.stochastic_rounding(override=False):
        out = ph0 + ph1
        outputs[out] = False
      with ipu.scopes.stochastic_rounding(override=True):
        out = ph0 + ph1
        outputs[out] = True
      with ipu.scopes.stochastic_rounding(override=False):
        with ipu.scopes.stochastic_rounding(override=True):
          out = ph0 + ph1
          outputs[out] = True
        out = ph0 + ph1
        outputs[out] = False
      with ipu.scopes.stochastic_rounding(override=True):
        with ipu.scopes.stochastic_rounding(override=False):
          out = ph0 + ph1
          outputs[out] = False
        out = ph0 + ph1
        outputs[out] = True

    with sl.Session() as sess:
      return sess.run(list(outputs.keys()), {
          ph0: np.full([4096], 1),
          ph1: np.full([4096], 1e-3)
      }), outputs

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_scoped_stochastic_rounding(self):
    self._configure(
        stochastic_rounding=ipu.config.StochasticRoundingBehaviour.ON)
    results, outputs = self._test_scoped_stochastic_rounding()
    for i, (_, expected_stochastic_rounding) in enumerate(outputs.items()):
      hist = np.histogram(results[i], bins=2)[0]
      if expected_stochastic_rounding:
        self.assertTrue(hist[0] > 0)
        self.assertTrue(hist[1] > 0)
      else:
        self.assertEqual(hist[0], 0)
        self.assertEqual(hist[1], 4096)

  @staticmethod
  def _test_nan_overflow():
    ph0 = array_ops.placeholder(np.float16, [1])
    ph1 = array_ops.placeholder(np.float16, [1])

    def model(x, y):
      return x * y

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(model, inputs=[ph0, ph1])

    with sl.Session() as sess:
      return sess.run(r, {ph0: [10000.], ph1: [10000.]})

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_nan_overflow(self):
    self._configure()
    self.assertEqual(self._test_nan_overflow(),
                     np.array([65500.], dtype=np.float16))

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_nan_overflow_enabled(self):
    self._configure(nan_overflow=True)
    self.assertTrue(np.isnan(self._test_nan_overflow()))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

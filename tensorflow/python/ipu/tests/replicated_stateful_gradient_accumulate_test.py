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

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops

from tensorflow.python import ipu


class TestReplicatedStatefulGradientAccumulate(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testStatefulGradientAccumulateAndCrossReplica(self):
    with tu.ipu_session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, _):
          return i < 10

        def body(i, y):
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(y), num_mini_batches=5, verify_usage=False)
          cr = gen_popops_ops.ipu_cross_replica_sum(ga)
          y = y + cr
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])
        report = gen_ipu_ops.ipu_event_trace()

      config = ipu.config.IPUConfig()
      config.optimizations.maximum_cross_replica_sum_buffer_size = 1000
      config.auto_select_ipus = [2]
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [30])

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCrossReplicaAndStatefulGradientAccumulate(self):
    with tu.ipu_session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, _):
          return i < 10

        def body(i, y):
          cr = gen_popops_ops.ipu_cross_replica_sum(array_ops.ones_like(y))
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              cr, num_mini_batches=5, verify_usage=False)
          y = y + ga
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])
        report = gen_ipu_ops.ipu_event_trace()

      config = ipu.config.IPUConfig()
      config.optimizations.maximum_cross_replica_sum_buffer_size = 1000
      config.auto_select_ipus = [2]
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [30])

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCrossReplicaAndNormalizeAndStatefulGradientAccumulate(self):
    with tu.ipu_session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, _):
          return i < 10

        def body(i, y):
          cr = gen_popops_ops.ipu_cross_replica_sum(array_ops.ones_like(y))
          norm = gen_poputil_ops.ipu_replication_normalise(cr)
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate(
              norm, num_mini_batches=5, verify_usage=False)
          y = y + ga
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])
        report = gen_ipu_ops.ipu_event_trace()

      config = ipu.config.IPUConfig()
      config.optimizations.maximum_cross_replica_sum_buffer_size = 1000
      config.auto_select_ipus = [2]
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [20])

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCrossReplicaAndNormalizeAndStatefulGradientAccumulateWithMom(self):
    with tu.ipu_session() as sess:
      dtype = np.float32

      def get_var():
        with variable_scope.variable_scope("",
                                           use_resource=True,
                                           reuse=variable_scope.AUTO_REUSE):
          var = variable_scope.get_variable(
              "x",
              dtype=dtype,
              shape=[1],
              initializer=init_ops.constant_initializer(0.0))
        return var

      def my_net(y):
        def cond(i, _):
          return i < 10

        def body(i, y):
          var = get_var()
          cr = gen_popops_ops.ipu_cross_replica_sum(
              array_ops.ones_like(y) * 2.0)
          norm = gen_poputil_ops.ipu_replication_normalise(cr)
          ga = gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
              var.handle, norm, momentum=0.8, num_mini_batches=5)
          y = y + ga
          i = i + 1
          return (i, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])
        report = gen_ipu_ops.ipu_event_trace()

      config = ipu.config.IPUConfig()
      config.optimizations.maximum_cross_replica_sum_buffer_size = 1000
      config.auto_select_ipus = [2]
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(variables.global_variables_initializer())
      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [38])
      v = sess.run(get_var())
      self.assertEqual(v, 18)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testMultiCrossReplicaAndNormalizeAndStatefulGradientAccumulateWithMom(
      self):
    with tu.ipu_session() as sess:
      dtype = np.float32

      def get_var(name):
        with variable_scope.variable_scope("",
                                           use_resource=True,
                                           reuse=variable_scope.AUTO_REUSE):
          var = variable_scope.get_variable(
              name,
              dtype=dtype,
              shape=[1],
              initializer=init_ops.constant_initializer(0.0))
        return var

      def my_net(y):
        def cond(i, _a, _b, _c):
          return i < 10

        def body(i, a, b, c):
          var_a = get_var("a")
          cr_a = gen_popops_ops.ipu_cross_replica_sum(
              array_ops.ones_like(a) * 2.5)
          norm_a = gen_poputil_ops.ipu_replication_normalise(cr_a)
          ga_a = gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
              var_a.handle, norm_a, momentum=0.8, num_mini_batches=5)
          a = a + ga_a

          # Note the different momentum value.
          var_b = get_var("b")
          cr_b = gen_popops_ops.ipu_cross_replica_sum(
              array_ops.ones_like(b) * 7.0)
          norm_b = gen_poputil_ops.ipu_replication_normalise(cr_b)
          ga_b = gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
              var_b.handle, norm_b, momentum=0.6, num_mini_batches=5)
          b = b + ga_b

          var_c = get_var("c")
          cr_c = gen_popops_ops.ipu_cross_replica_sum(
              array_ops.ones_like(c) * 8.0)
          norm_c = gen_poputil_ops.ipu_replication_normalise(cr_c)
          ga_c = gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
              var_c.handle, norm_c, momentum=0.8, num_mini_batches=5)
          c = c + ga_c
          i = i + 1
          return (i, a, b, c)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])
        report = gen_ipu_ops.ipu_event_trace()

      config = ipu.config.IPUConfig()
      config.optimizations.maximum_cross_replica_sum_buffer_size = 1000
      config.auto_select_ipus = [2]
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(variables.global_variables_initializer())
      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [45])
      self.assertAllClose(y[2], [101])
      self.assertAllClose(y[3], [122])
      v = sess.run(get_var("a"))
      self.assertEqual(v, 22.5)
      v = sess.run(get_var("b"))
      self.assertEqual(v, 56)
      v = sess.run(get_var("c"))
      self.assertEqual(v, 72)


if __name__ == "__main__":
  googletest.main()

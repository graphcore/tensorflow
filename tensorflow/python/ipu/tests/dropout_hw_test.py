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
from tensorflow.python.client import session as sl
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python import ipu
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops


# Call the gen op directly in order to access the seed.
def _dropout(x, noise_shape=None):
  return gen_poprand_ops.ipu_dropout(x,
                                     rate=0.5,
                                     scale=1.0,
                                     noise_shape=noise_shape)


class DropoutTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.deprecated_graph_mode_only
  def testDropoutInPipeline(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[32])
      return dataset.batch(batch_size=32, drop_remainder=True)

    pipeline_depth = 20
    repeat_count = 2

    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[32, 32],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x = math_ops.matmul(x, weight)
      x, s1, _ = _dropout(x)
      return x, s1

    def stage2(x, s1):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[32, 32],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x = math_ops.matmul(x, weight)
      x, s2, _ = _dropout(x)
      return x, s1, s2

    def stage3(x, s1, s2):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[32, 32],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x, s3, _ = _dropout(x)
      x = math_ops.matmul(x, weight)
      return x, s1, s2, s3

    def stage4(x, s1, s2, s3):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w3",
            shape=[32, 32],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x = math_ops.matmul(x, weight)
      x, s4, _ = _dropout(x)
      return x, s1, s2, s3, s4

    def stage5(x, s1, s2, s3, s4):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w4",
            shape=[32, 32],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x = math_ops.matmul(x, weight)
      _, s5, _ = _dropout(x, noise_shape=[32, 1])
      return s1, s2, s3, s4, s5

    output = pipelining_test_util.PipelineTester.pipeline_on_ipu(
        [stage1, stage2, stage3, stage4, stage5],
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        None,
        self,
        10000,
        recomp=False,
        schedule=ipu.pipelining_ops.PipelineSchedule.Grouped,
        device_mapping=[0, 2, 1, 3, 1],
        replication_factor=2)

    s1, s2, s3, s4, s5 = output
    for seeds in (s1, s2, s3, s4, s5):
      assert len(seeds) == pipeline_depth * repeat_count
      for i, s in enumerate(seeds):
        # Make sure the seeds are different between the replicas.
        self.assertNotAllEqual(s[0], s[1])
        # Make sure the seeds between iterations are different.
        self.assertNotAllEqual(s, seeds[i - 1])
        # Make sure the seeds are different between repeat counts.
        self.assertNotAllEqual(s, seeds[i - pipeline_depth])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testMultipleExecutionsNoUserSeed(self):
    def model(x):
      return gen_poprand_ops.ipu_dropout(x, rate=0.5, scale=1.0)[0:2]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [2], name="data")

    with ops.device("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(model, [inp])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      _, s1 = sess.run(out, {inp: [2, 4]})
      _, s2 = sess.run(out, {inp: [2, 4]})
      self.assertNotAllEqual(s1, s2)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testMultipleExecutionsUserSeed(self):
    def model(x):
      return gen_poprand_ops.ipu_dropout_with_seed(x, [10, 10],
                                                   rate=0.5,
                                                   scale=1.0)[0:2]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [2], name="data")

    with ops.device("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(model, [inp])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      _, s1 = sess.run(out, {inp: [2, 4]})
      _, s2 = sess.run(out, {inp: [2, 4]})
      self.assertAllEqual(s1, s2)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testFwdBwdSeedMatches(self):
    def model():
      with variable_scope.variable_scope("vs", use_resource=True):
        w = variable_scope.get_variable(
            "w",
            shape=[2, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())

      loss, s, _ = gen_poprand_ops.ipu_dropout(w, rate=0.5, scale=1.0)
      g = gradients_impl.gradients(loss, [w])
      assert len(g) == 1
      dropout_grad = g[0]
      assert dropout_grad.op.type == "IpuDropoutWithSeedAndReference"
      return s, dropout_grad.op.outputs[1]

    with ops.device("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(model, [])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())
      s_fwd, s_bwd = sess.run(out)
      self.assertAllEqual(s_fwd, s_bwd)
      s_fwd, s_bwd = sess.run(out)
      self.assertAllEqual(s_fwd, s_bwd)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testReuseSequence(self):
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def inner_loop(x):
      @ipu.outlined_function
      def f(z):
        return gen_poprand_ops.ipu_dropout(z, rate=0.5, scale=1.0)

      x, s1, _ = f(x)
      x, s2, _ = f(x)
      return x, outfeed_queue.enqueue([s1, s2])

    def model(x):
      return ipu.loops.repeat(10, inner_loop, [x])

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [2], name="data")

    with ops.device("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(model, [inp])

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    outfeed_op = outfeed_queue.dequeue()

    with sl.Session() as sess:
      sess.run(out, {inp: [2, 4]})
      s1, s2 = sess.run(outfeed_op)
      seeds = list(zip(s1, s2))
      for i, s in enumerate(seeds):
        # Make sure the seeds are not the between the executions of the same
        # function.
        self.assertNotAllEqual(s[0], s[1])
        # Make sure the seeds between iterations are different.
        self.assertNotAllEqual(s, seeds[i - 1])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testDropoutTrainingInPipeline(self):
    def dataset_fn():
      transform = np.roll(np.eye(100), 1, 1)
      inputs = np.random.randn(1000, 100) / np.sqrt(100)
      outputs = inputs @ transform
      return Dataset.from_tensor_slices(
          dict(
              inputs=inputs.astype(np.float32),
              outputs=outputs.astype(np.float32),
          )).repeat().batch(32, drop_remainder=True)

    repeat_count = 32
    pipeline_depth = 32

    def stage1(inputs, outputs):
      N = int(inputs.shape[-1])
      x = inputs @ variable_scope.get_variable(
          "w0",
          shape=(N, N),
          dtype=np.float32,
          initializer=init_ops.zeros_initializer())
      x = ipu.rand_ops.dropout(x)
      return x, outputs

    def stage2(inputs, outputs):
      l = math_ops.reduce_mean(
          math_ops.reduce_sum((outputs - inputs)**2, axis=-1))
      return l

    output = pipelining_test_util.PipelineTester.pipeline_on_ipu(
        [stage1, stage2],
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        gradient_descent.GradientDescentOptimizer(0.1),
        self,
        10000,
        recomp=True,
        device_mapping=[0, 3],
        schedule=ipu.pipelining_ops.PipelineSchedule.Grouped)

    assert output[-1] < output[0]

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testDropoutTrainingInPipelineHighRate(self):
    def dataset_fn():
      transform = np.roll(np.eye(100), 1, 1)
      inputs = np.random.randn(1000, 100) / np.sqrt(100)
      outputs = inputs @ transform
      return Dataset.from_tensor_slices(
          dict(
              inputs=inputs.astype(np.float32),
              outputs=outputs.astype(np.float32),
          )).repeat().batch(32, drop_remainder=True)

    repeat_count = 32
    pipeline_depth = 32

    def stage1(inputs, outputs):
      N = int(inputs.shape[-1])
      x = inputs @ variable_scope.get_variable(
          "w0",
          shape=(N, N),
          dtype=np.float32,
          initializer=init_ops.zeros_initializer())
      x = ipu.rand_ops.dropout(x, rate=0.9)
      return x, outputs

    def stage2(inputs, outputs):
      l = math_ops.reduce_mean(
          math_ops.reduce_sum((outputs - inputs)**2, axis=-1))
      return l

    output = pipelining_test_util.PipelineTester.pipeline_on_ipu(
        [stage1, stage2],
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        gradient_descent.GradientDescentOptimizer(0.1),
        self,
        10000,
        recomp=True,
        device_mapping=[0, 3],
        schedule=ipu.pipelining_ops.PipelineSchedule.Grouped)

    assert output[-1] < output[0]


if __name__ == "__main__":
  googletest.main()

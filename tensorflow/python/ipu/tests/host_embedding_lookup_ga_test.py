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

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent as gd
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ipu import loops

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


class HostEmbeddingLookupGATest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testDIENShape(self):
    shape = [10000000, 20]  # 740MB at float32
    lookup_count = 4096

    def my_net(i):

      # lookup
      out = gen_pop_datastream_ops.ipu_device_embedding_lookup(
          i,
          embedding_id="host_embedding",
          embedding_shape=shape,
          dtype=np.float32)

      # update
      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      # notify
      gen_pop_datastream_ops.ipu_device_embedding_notify(
          embedding_id="host_embedding")

      self.assertEqual(out.shape, (lookup_count, shape[1]))
      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])
      w = variable_scope.get_variable("foo",
                                      dtype=np.float32,
                                      shape=shape,
                                      use_resource=False)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)

      sess.run(variables.global_variables_initializer())
      report.reset()
      sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_register(
              w, "host_embedding", optimizer="SGD+GA"))
      result = sess.run([r], {i: i_h})
      v = sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_deregister(
              w, "host_embedding"))

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 3, np.take(v, i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))
      report.parse_log()
      report.assert_max_tile_memory(772, tolerance=0.3)

  @test_util.deprecated_graph_mode_only
  def testAGIShape(self):
    shape = [100000, 200]
    lookup_count = 4096

    def my_net(i):
      # lookup
      out = gen_pop_datastream_ops.ipu_device_embedding_lookup(
          i,
          embedding_id="host_embedding",
          embedding_shape=shape,
          dtype=np.float32)

      # update
      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      gen_pop_datastream_ops.ipu_device_embedding_update_add(
          out, out, i, embedding_id="host_embedding", embedding_shape=shape)

      # notify
      gen_pop_datastream_ops.ipu_device_embedding_notify(
          embedding_id="host_embedding")

      self.assertEqual(out.shape, (lookup_count, shape[1]))
      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])
      w = variable_scope.get_variable("foo",
                                      dtype=np.float32,
                                      shape=shape,
                                      use_resource=False)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()
      sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_register(
              w, "host_embedding", optimizer="SGD+GA"))
      result = sess.run([r], {i: i_h})
      v = sess.run(
          gen_pop_datastream_ops.ipu_host_embedding_deregister(
              w, "host_embedding"))

      # Since we updated with the same activations, we expect to see a 2x
      self.assertAllClose(result[0][0] * 3, np.take(v, i_h, axis=0))
      self.assertEqual(result[0][0].shape, (lookup_count, shape[1]))
      report.parse_log()
      report.assert_max_tile_memory(5852, tolerance=0.3)

  @test_util.deprecated_graph_mode_only
  def testTrainNoExec(self):
    shape = [100000, 200]
    lookup_count = 4096

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingSGDGAOptimizerSpec(0.5, 2))

    def my_net(i):
      out = host_embedding.lookup(i)

      return out

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.utils.create_ipu_config(profiling=True,
                                      always_rearrange_copies_on_the_host=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()

      with host_embedding.register(sess):
        # training=False should ignore the number of expected updates.
        result = sess.run([r], {i: i_h})

      v = sess.run(host_embedding.get_embedding_tensor())
      # Check the lookup result, but we are really interested that it doesn't hang.
      self.assertAllClose(result[0][0], np.take(v, i_h, axis=0))

  @test_util.deprecated_graph_mode_only
  def testNoLookup(self):
    shape = [100000, 200]
    lookup_count = 4096

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingSGDGAOptimizerSpec(0.5, 2))

    def my_net(i):
      return i

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.utils.create_ipu_config(profiling=True,
                                      always_rearrange_copies_on_the_host=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()

      with host_embedding.register(sess):
        result = sess.run([r], {i: i_h})

      # Check the indices are correct, but the real test is no timeout.
      self.assertAllClose(result[0][0], i_h)

  @test_util.deprecated_graph_mode_only
  def testModel(self):
    shape = [1000, 256]
    lookup_count = 128
    lr = 1 / 2
    acc_factor = 2
    num_iterations = 6

    host_embedding = embedding_ops.create_host_embedding(
        "my_host_embedding",
        shape,
        np.float32,
        optimizer_spec=embedding_ops.HostEmbeddingSGDGAOptimizerSpec(
            lr, acc_factor))

    optimizer = ga.GradientAccumulationOptimizerV2(
        gd.GradientDescentOptimizer(lr), acc_factor)

    # A dummy model that has an embedding lookup and a matmul
    def model(i, w):
      a = host_embedding.lookup(i)
      return math_ops.matmul(a, w)

    def training(loss, i, w):
      loss = model(i, w)
      mean_loss = math_ops.reduce_mean(loss)
      abs_mean_loss = math_ops.abs(mean_loss)
      train = optimizer.minimize(abs_mean_loss)
      return mean_loss, i, w, train

    def my_net(i, w):
      loss = array_ops.constant(0.0, shape=[])
      r = loops.repeat(num_iterations, training, [loss, i, w])
      return r

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [lookup_count])
      w = array_ops.placeholder(np.float32, [256, 128])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i, w])

    cfg = ipu.utils.create_ipu_config(profiling=True,
                                      always_rearrange_copies_on_the_host=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, lookup_count).reshape([lookup_count])
      w_h = np.random.rand(256, 128).astype(np.float32)

      report = tu.ReportJSON(self, sess, configure_device=False)
      sess.run(variables.global_variables_initializer())
      report.reset()

      with host_embedding.register(sess):
        result = sess.run([r], {i: i_h, w: w_h})

      # Given the dumb model and the LR is the inverse of the accumulation factor,
      # we expect the "mean loss" to be zero.
      self.assertAllClose(result[0][0], 0.0)


if __name__ == "__main__":
  googletest.main()

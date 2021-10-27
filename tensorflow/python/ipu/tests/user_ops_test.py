#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

import os
import pva
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


def count_grad_ops(graph):
  num_grad_ops = 0
  for op in graph.get_operations():
    if op.type == "IpuUserOp" and op.get_attr("gradient_size") > 0:
      num_grad_ops = num_grad_ops + 1
  return num_grad_ops


class UserProvidedOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testUserOp(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32, dtypes.float32, dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
              tensor_shape.TensorShape([5, 2]),
              tensor_shape.TensorShape([10])
          ],
      }
      lib_path = cwd + "/tensorflow/python/ipu/libadd_incrementing_custom.so"

      def my_net(x, y, z):
        o1 = ipu.custom_ops.precompiled_user_op([x, y, z],
                                                lib_path,
                                                outs=outputs)

        o2 = ipu.custom_ops.precompiled_user_op([x + 1., y + 1., z + 1.],
                                                lib_path,
                                                outs=outputs)
        return o1, o2

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[20])
        y = array_ops.placeholder(np.float32, shape=[5, 2])
        z = array_ops.placeholder(np.float32, shape=[10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([20]),
          y: np.ones([5, 2]),
          z: np.ones([10])
      })

      self.assertAllEqual(np.full([20], 2.0), res[0][0])
      self.assertAllEqual(np.full([5, 2], 3.0), res[0][1])
      self.assertAllEqual(np.full([10], 4.0), res[0][2])
      self.assertAllEqual(np.full([20], 3.0), res[1][0])
      self.assertAllEqual(np.full([5, 2], 4.0), res[1][1])
      self.assertAllEqual(np.full([10], 5.0), res[1][2])

  @test_util.deprecated_graph_mode_only
  def testUserOpWithAllocate(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([128])],
      }

      lib_path = os.path.join(
          cwd,
          "tensorflow/python/ipu/libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y):
        x = ipu.custom_ops.precompiled_user_op([x, y],
                                               lib_path,
                                               op_name="AllocTest",
                                               outs=outputs)
        return x

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[128])
        y = array_ops.placeholder(np.float32, shape=[128])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([128]),
          y: np.ones([128]),
      })

      report_json.parse_log()

      found = 0
      for t in report_json.get_tensor_map().all_tensors():
        if t.inst == "arg0.1":
          # Allocator maps all of input 0 to tile 0
          self.assertAllEqual(t.tile_ids(), [0])
          found = found + 1
        if t.inst == "arg1.2":
          # Allocator leaves input 1 to be linearly mapped
          self.assertAllEqual(t.tile_ids(), [0, 1, 2, 3])
          found = found + 1

      self.assertAllEqual(found, 2)
      self.assertAllEqual(np.full([128], 2.0), res[0])

  def runCustomUserOpWithUnusedOutput(self, op_name, ok):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([128])],
      }

      lib_path = os.path.join(
          cwd,
          "tensorflow/python/ipu/libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y):
        ipu.custom_ops.precompiled_user_op([x, y],
                                           lib_path,
                                           op_name=op_name,
                                           outs=outputs)
        return [x + y]

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[128])
        y = array_ops.placeholder(np.float32, shape=[128])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      sess.run(variables.global_variables_initializer())
      sess.run(model, {
          x: np.ones([128]),
          y: np.ones([128]),
      })

      report = pva.openReport(report_helper.find_report())
      self.assert_all_compute_sets_and_list(report, ok)

  @test_util.deprecated_graph_mode_only
  def testStatefulUserOp(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    ok = [
        '__seed*',
        'add/add.*/Op/Add',
        'Stateful/Op/Add',
    ]
    return self.runCustomUserOpWithUnusedOutput("Stateful", ok)

  @test_util.deprecated_graph_mode_only
  def testStatelessUserOp(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    ok = [
        '__seed*',
        'add/add.*/Op/Add',
    ]
    return self.runCustomUserOpWithUnusedOutput("Stateless", ok)

  @test_util.deprecated_graph_mode_only
  def testUserOpBackwards(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32, dtypes.float32, dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
              tensor_shape.TensorShape([5, 2]),
              tensor_shape.TensorShape([10])
          ],
      }
      lib_path = cwd + "/tensorflow/python/ipu/libadd_incrementing_custom.so"

      def my_net(x, y, z):
        output = ipu.custom_ops.precompiled_user_op([x, y, z],
                                                    lib_path,
                                                    outs=outputs)

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        gradients = opt.compute_gradients(output[2], [x, y, z])

        return [output, gradients]

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[20])
        y = array_ops.placeholder(np.float32, shape=[5, 2])
        z = array_ops.placeholder(np.float32, shape=[10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

      self.assertAllEqual(count_grad_ops(ops.get_default_graph()), 1)

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([20]),
          y: np.ones([5, 2]),
          z: np.ones([10])
      })

      inputs = res[0]

      self.assertAllEqual(np.full([20], 2.0), inputs[0])
      self.assertAllEqual(np.full([5, 2], 3.0), inputs[1])
      self.assertAllEqual(np.full([10], 4.0), inputs[2])

      gradients = res[1]

      # Our gradient function is the same as the above but a multiply instead.
      # Since the "loss" is just output[3], input[3] is the only one which
      # will actually have a gradient. (Which will be 3).
      self.assertAllEqual(np.zeros([20]), gradients[0][0])
      self.assertAllEqual(np.zeros([5, 2]), gradients[1][0])
      self.assertAllEqual(np.full([10], 3.0), gradients[2][0])

  @test_util.deprecated_graph_mode_only
  def testUserReadWriteOpBackwardsUnusedGradients(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    SIZE = 5

    def scaled_add_op(x, scale, y):
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([SIZE])],
      }
      base_dir = os.path.join(cwd, "tensorflow/python/ipu")
      gp_path = os.path.join(base_dir,
                             "tests/add_scaled_vector_add_codelet.cc")
      lib_path = os.path.join(base_dir, "libadd_partial_gradients_custom.so")

      return ipu.custom_ops.precompiled_user_op(
          [x, scale, y, math_ops.cos(x),
           math_ops.cosh(y)],
          lib_path,
          gp_path,
          outs=outputs,
          inputs_with_gradients=[0, 2])

    def model(scale, y, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        x = variable_scope.get_variable(
            "x",
            shape=[SIZE],
            initializer=init_ops.ones_initializer(),
            dtype=np.float32)
      z = math_ops.reduce_mean(scaled_add_op(x, scale, y), axis=1)
      loss = losses.mean_squared_error(label, z)
      return loss, gradient_descent.GradientDescentOptimizer(0.01).minimize(
          loss)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      scale_data = array_ops.placeholder(np.float32, [])
      y_data = array_ops.placeholder(np.float32, [SIZE])
      label_data = array_ops.placeholder(np.int32, [1])

      xla_result = ipu.ipu_compiler.compile(model,
                                            [scale_data, y_data, label_data])

    with tu.ipu_session() as sess:
      scale = 2
      b = np.full([SIZE], 3)
      label = np.ones([1])
      sess.run(variables.global_variables_initializer())

      result = sess.run(xla_result,
                        feed_dict={
                            y_data: b,
                            scale_data: scale,
                            label_data: label
                        })

      self.assertEqual(result[0], 36)

  @test_util.deprecated_graph_mode_only
  def testUserReadWriteOpBackwards(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [tensor_shape.TensorShape([10])],
      }
      lib_path = cwd + "/tensorflow/python/ipu/libadd_tensors_custom.so"

      def my_net(x, y):
        output = ipu.custom_ops.cpu_user_operation([x, y],
                                                   lib_path,
                                                   outs=outputs)

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)
        gradients = opt.compute_gradients(output[0], [x, y])

        return [output, gradients]

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[10])
        y = array_ops.placeholder(np.float32, shape=[10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([10]),
          y: np.full([10], 6.0),
      })

      self.assertAllEqual(np.full([1, 10], 7.0), res[0])

      gradients = res[1]
      self.assertAllEqual(np.ones([10]), gradients[0][0])

  @test_util.deprecated_graph_mode_only
  def testUserOpBackwardsSeparateOps(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32, dtypes.float32, dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
              tensor_shape.TensorShape([5, 2]),
              tensor_shape.TensorShape([10])
          ],
      }

      lib_path = os.path.join(
          cwd,
          "tensorflow/python/ipu/libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y, z):
        output = ipu.custom_ops.precompiled_user_op([x, y, z],
                                                    lib_path,
                                                    op_name="SepGrad",
                                                    separate_gradients=True,
                                                    outs=outputs)
        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        gradients = opt.compute_gradients(output[2], [x, y, z])

        return [output, gradients]

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[20])
        y = array_ops.placeholder(np.float32, shape=[5, 2])
        z = array_ops.placeholder(np.float32, shape=[10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

      self.assertAllEqual(count_grad_ops(ops.get_default_graph()), 3)

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([20]),
          y: np.ones([5, 2]),
          z: np.ones([10])
      })

      inputs = res[0]

      self.assertAllEqual(np.full([20], 2.0), inputs[0])
      self.assertAllEqual(np.full([5, 2], 3.0), inputs[1])
      self.assertAllEqual(np.full([10], 4.0), inputs[2])

      gradients = res[1]

      # The grad function adds index+1 to the value of the partial derivative
      # index. Since the "loss" is just output[2], input[2] is the only one
      # which will actually have a gradient. (Which will be 1*3 = 3).
      self.assertAllEqual(np.zeros([20]), gradients[0][0])
      self.assertAllEqual(np.zeros([5, 2]), gradients[1][0])
      self.assertAllEqual(np.full([10], 3.0), gradients[2][0])

  # We test this one with a different SO to implicitly test what happens if
  # we don't have metadata in the above tests.
  @test_util.deprecated_graph_mode_only
  def testUserOpMetadata(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32, dtypes.float32, dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
              tensor_shape.TensorShape([5, 2]),
              tensor_shape.TensorShape([10])
          ],
      }

      lib_path = os.path.join(
          cwd,
          "tensorflow/python/ipu/libadd_incrementing_custom_with_metadata.so")

      def my_net(x, y, z):
        output = ipu.custom_ops.precompiled_user_op([x, y, z],
                                                    lib_path,
                                                    outs=outputs)

        opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)

        gradients = opt.compute_gradients(output[2], [x, y, z])

        return [output, gradients]

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[20])
        y = array_ops.placeholder(np.float32, shape=[5, 2])
        z = array_ops.placeholder(np.float32, shape=[10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([20]),
          y: np.ones([5, 2]),
          z: np.ones([10])
      })

      inputs = res[0]

      self.assertAllEqual(np.full([20], 2.0), inputs[0])
      self.assertAllEqual(np.full([5, 2], 3.0), inputs[1])
      self.assertAllEqual(np.full([10], 4.0), inputs[2])

      gradients = res[1]

      # Our gradient function is the same as the above but a multiply
      # instead. Since the "loss" is just output[3], input[3] is the only
      # one which will actually have a gradient. (Which will be 3).
      self.assertAllEqual(np.zeros([20]), gradients[0][0])
      self.assertAllEqual(np.zeros([5, 2]), gradients[1][0])
      self.assertAllEqual(np.full([10], 3.0), gradients[2][0])

  @test_util.deprecated_graph_mode_only
  def testUserOpCPU(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32, dtypes.int32, dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
              tensor_shape.TensorShape([10, 10, 10]),
              tensor_shape.TensorShape([1]),
          ],
      }
      lib_path = cwd + "/tensorflow/python/ipu/libadd_incrementing_custom.so"

      def my_net(x, y):
        output = ipu.custom_ops.cpu_user_operation([x, y],
                                                   lib_path,
                                                   outs=outputs)
        return output

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        x = array_ops.placeholder(np.float32, shape=[20])
        y = array_ops.placeholder(np.int32, shape=[10, 10, 10])

        model = ipu.ipu_compiler.compile(my_net, inputs=[x, y])

      sess.run(variables.global_variables_initializer())
      res = sess.run(
          model, {
              x: np.ones([20]),
              y: np.full([10, 10, 10], fill_value=6, dtype=np.int32),
          })

      # The first operation is in[0] + 6
      self.assertAllEqual(np.full([20], 7.0), res[0])

      # The second part is in[1] / 2
      self.assertAllEqual(np.full([10, 10, 10], 3, dtype=np.int32), res[1])

      # The third part is the sum of the last two so 20*7 + 1000*3.
      self.assertAllEqual(np.full([1], 3140.0), res[2])

  @test_util.deprecated_graph_mode_only
  def testUserOpLoadNonExistentSharedLibrary(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
          ],
      }
      lib_path = cwd + "/and-now-for-something-completely-different.so"

      def my_net(x):
        return ipu.custom_ops.precompiled_user_op([x], lib_path, outs=outputs)

      with self.assertRaises(errors_impl.NotFoundError):
        with ipu.scopes.ipu_scope('/device:IPU:0'):
          x = array_ops.placeholder(np.float32, shape=[20])
          model = ipu.ipu_compiler.compile(my_net, inputs=[x])

        sess.run(variables.global_variables_initializer())
        sess.run(model, {
            x: np.ones([20]),
        })

  @test_util.deprecated_graph_mode_only
  def testUserOpLoadLibraryWithWrongApiLevel(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:
      cwd = os.getcwd()
      outputs = {
          "output_types": [dtypes.float32],
          "output_shapes": [
              tensor_shape.TensorShape([20]),
          ],
      }
      lib_path = cwd + "/tensorflow/python/ipu/libwrong_api_level_custom.so"

      def my_net(x):
        return ipu.custom_ops.precompiled_user_op([x], lib_path, outs=outputs)

      with self.assertRaises(errors_impl.InternalError):
        with ipu.scopes.ipu_scope('/device:IPU:0'):
          x = array_ops.placeholder(np.float32, shape=[20])
          model = ipu.ipu_compiler.compile(my_net, inputs=[x])

        sess.run(variables.global_variables_initializer())
        sess.run(model, {
            x: np.ones([20]),
        })


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

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
# =============================================================================

import numpy as np
import pva

from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu.config import SchedulingAlgorithm
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util_v2 as control_util
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class FunctionalOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testFunctionInferenceWithVariableScope(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def func(a, b, name):
        @ipu.outlined_function
        def outlined_func(a, b):
          with variable_scope.variable_scope(name, use_resource=True):
            w = variable_scope.get_variable(
                "w",
                shape=[64, 64],
                dtype=np.float32,
                initializer=init_ops.ones_initializer())
          x = math_ops.matmul(a, w)
          x = x + b
          return math_ops.sigmoid(x)

        return outlined_func(a, b)

      def body(a, b, c):
        a = func(a, b, name="one")
        a = a - func(a, c, name="two")
        return a

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [64, 64])
        b = array_ops.placeholder(np.float32, [64, 64])
        c = array_ops.placeholder(np.float32, [64, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b, c])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report_json.parse_log()
      # Entry computation and outlined one.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 2)

    report = pva.openReport(report_helper.find_report())
    # There would be multiple non-linearities if the function was not
    # cached.
    ok = [
        'MatMul/dot*/Conv_1',
        'add/add*/Op/Add',
        'Sigmoid/sigmoid/Nonlinearity',
        'sub/subtract*/Op/Subtract',
    ]
    self.assert_all_compute_sets_and_list(report, ok)
    self.assert_total_tile_memory(report, 204616, tolerance=0.1)
    self.assert_max_tile_memory(report, 25647, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testFunctionTraining(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(lhs, rhs, a):
        x = math_ops.matmul(lhs, rhs)
        x = x + a
        x = math_ops.sigmoid(x)
        return x

      def body(a, b, c, labels):
        with variable_scope.variable_scope("vs", use_resource=True):
          w0 = variable_scope.get_variable(
              "w0",
              shape=[64, 64],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
          w1 = variable_scope.get_variable(
              "w1",
              shape=[64, 64],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
        a = func(a, w0, b)
        a = a - func(a, w1, c)
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=a,
                                                        labels=labels))
        train_op = gradient_descent.GradientDescentOptimizer(0.001).minimize(
            loss)
        return a, train_op

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [64, 64])
        b = array_ops.placeholder(np.float32, [64, 64])
        c = array_ops.placeholder(np.float32, [64, 64])
        labels = array_ops.placeholder(np.int32, [64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b, c, labels])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c, labels]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report_json.parse_log()

      # Entry computastion and 2 outlined ones.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 3)

    report = pva.openReport(report_helper.find_report())
    # There would be multiple non-linearities(grads) if the function was not
    # cached.
    # pylint: disable=line-too-long
    ok = [
        'MatMul/dot*/Conv_1', 'add/add*/Op/Add',
        'Sigmoid/sigmoid/Nonlinearity', 'sub/subtract*/Op/Subtract',
        'SparseSoftmaxCrossEntropyWithLogits',
        'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul',
        'gradients/sub_grad/Neg/negate*/Op/Negate',
        'gradients/Sigmoid_grad/SigmoidGrad/sigmoid-grad*/NonLinearityGrad',
        'gradients/AddN/scaled-inplace',
        'GradientDescent/update_vs/w*/ResourceApplyGradientDescent/scaled-inplace',
        'gradients/MatMul_grad/MatMul_1/dot', '/Transpose'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)
    self.assert_total_tile_memory(report, 322656, tolerance=0.1)
    self.assert_max_tile_memory(report, 40691, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testNestedFunctionTraining(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def matmul_with_bias(x, scope_name):
        @ipu.outlined_function
        def func(x):
          with variable_scope.variable_scope(scope_name, use_resource=True):
            w = variable_scope.get_variable(
                "w",
                shape=[64, 64],
                dtype=np.float32,
                initializer=init_ops.ones_initializer())
          x = x @ w
          with variable_scope.variable_scope(scope_name, use_resource=True):
            bias = variable_scope.get_variable(
                "bias",
                shape=[x.shape.as_list()[-1]],
                dtype=np.float32,
                initializer=init_ops.ones_initializer())
          return x + bias

        return func(x)

      def cached_func(x, scope_name):
        @ipu.outlined_function
        def func(x):
          x = matmul_with_bias(x, scope_name)
          x = math_ops.sigmoid(x)
          return x

        return func(x)

      def body(x, labels):
        x = cached_func(x, "1")
        x = cached_func(x, "2")
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=labels))
        train_op = gradient_descent.GradientDescentOptimizer(0.001).minimize(
            loss)
        return x, train_op

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [64, 64])
        labels = array_ops.placeholder(np.int32, [64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, labels])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, labels]})
      self.assertAllClose(result[0], np.broadcast_to(1., [64, 64]))

      report_json.parse_log()

      # Entry computastion and 4 outlined ones.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 5)

    report = pva.openReport(report_helper.find_report())
    # There would be multiple non-linearities(grads) if the function was not
    # cached.
    # pylint: disable=line-too-long
    ok = [
        'matmul/dot*/Conv_1',
        'add_0/fusion/Op/Add',
        'Sigmoid/sigmoid/Nonlinearity',
        'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits',
        'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/',
        'gradients/Sigmoid_grad/SigmoidGrad/sigmoid-grad/NonLinearityGrad',
        'gradients/add_grad/Sum/reduce*/Reduce',
        'GradientDescent/update_1/bias/ResourceApplyGradientDescent/scaled-inplace',
        'GradientDescent/update_1/w/ResourceApplyGradientDescent/scaled-inplace',
        'GradientDescent/update_2/bias/ResourceApplyGradientDescent/scaled-inplace',
        'GradientDescent/update_2/w/ResourceApplyGradientDescent/scaled-inplace',
        '/Transpose',
        'gradients/matmul_grad/MatMul/dot',
        'gradients/matmul_grad/MatMul_1/dot',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)
    self.assert_total_tile_memory(report, 299632, tolerance=0.1)
    self.assert_max_tile_memory(report, 38837, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testFunctionSerializedLookup(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.scheduling.algorithm = SchedulingAlgorithm.POST_ORDER
    cfg.configure_ipu_system()

    # Disable the arithmetic_optimization Grappler pass, as it combines the
    # slice additions in this test into an AddN which gives them all the same
    # name, which means we can't look for them individually in the compute sets.
    with tu.ipu_session(
        disable_grappler_optimizers=["arithmetic_optimization"]) as sess:

      @ipu.outlined_function(keep_input_layouts=False)
      def func(table, indices, min_idx, max_idx):
        # Do a serialized embedding lookup by adjusting the indices.
        adjusted_indices = indices - min_idx
        x = ipu.embedding_ops.embedding_lookup(table, adjusted_indices)
        # Mask out any outputs which are not in range [min_idx, max_idx).
        mask_max = math_ops.less(indices, max_idx)
        mask_min = math_ops.greater_equal(indices, min_idx)
        mask = math_ops.cast(math_ops.logical_and(mask_max, mask_min),
                             np.float16)
        mask = array_ops.expand_dims(mask, 1)
        return x * mask

      DICT_SIZE = 20000
      EMB_SIZE = 128
      NUM_SPLITS = 10
      SPLIT_SIZE = DICT_SIZE // NUM_SPLITS

      def body(table, indices):
        table_sliced = array_ops.slice(table, [0, 0], [SPLIT_SIZE, EMB_SIZE])
        output = func(table_sliced, indices, 0, SPLIT_SIZE)

        for i in range(1, NUM_SPLITS):
          min_idx = SPLIT_SIZE * i
          max_idx = SPLIT_SIZE * (i + 1)
          table_sliced = array_ops.slice(table, [min_idx, 0],
                                         [SPLIT_SIZE, EMB_SIZE])
          output = math_ops.add(output,
                                func(table_sliced, indices, min_idx, max_idx),
                                name=f"slice_{i}")
        return output

      with ops.device('cpu'):
        table = array_ops.placeholder(np.float16, [DICT_SIZE, EMB_SIZE])
        indices = array_ops.placeholder(np.int32, [NUM_SPLITS * 2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[table, indices])

      report_json = tu.ReportJSON(self, sess)
      i_h = np.arange(0, DICT_SIZE, step=SPLIT_SIZE // 2)
      w_h = np.arange(EMB_SIZE, dtype=np.float16) * np.ones(
          [DICT_SIZE, EMB_SIZE], dtype=np.float16)
      result = sess.run(res, {table: w_h, indices: i_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))

      report_json.parse_log()

      # Main computation and outlined serialized one.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 2)

    report = pva.openReport(report_helper.find_report())
    # There would be multiple multi slices if the function was not cached.
    ok = [
        'Less/fusion*/Op/LessThan',
        'GreaterEqual/fusion*/Op/GreaterThanEqual',
        'sub/fusion/Op/Subtract',
        'embedding_lookup/multi-slice/output/multiSlice',
        'LogicalAnd/and*/Op/LogicalAnd',
        'Cast/convert*/Cast',
        'mul_0/fusion*/Op/Multiply',
        'slice_1*/add.*/Op/Add',
        'slice_2*/add.*/Op/Add',
        'slice_3*/add.*/Op/Add',
        'slice_4*/add.*/Op/Add',
        'slice_5*/add.*/Op/Add',
        'slice_6*/add.*/Op/Add',
        'slice_7*/add.*/Op/Add',
        'slice_8*/add.*/Op/Add',
        'slice_9*/add.*/Op/Add',
    ]
    self.assert_all_compute_sets_and_list(report, ok)
    self.assert_total_tile_memory(report, 6415442, tolerance=0.1)
    self.assert_max_tile_memory(report, 802278, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testFunctionsNoMatch(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(a):
        return nn.relu(a)

      def body(a, b, c):
        return func(a), func(b), func(c)

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float16, [64, 64])
        b = array_ops.placeholder(np.float16, [64, 64])
        c = array_ops.placeholder(np.float32, [64, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b, c])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c]})
      self.assertAllClose(result[0], np.broadcast_to(1.0, [64, 64]))
      self.assertAllClose(result[1], np.broadcast_to(1.0, [64, 64]))
      self.assertAllClose(result[2], np.broadcast_to(1.0, [64, 64]))

      report_json.parse_log()

      # Main computation (including inlined fp32 one, and the fp16 outlined).
      self.assertEqual(len(report_json.tensor_map.computation_names()), 2)

    report = pva.openReport(report_helper.find_report())
    # Two non-linearties, as one of them has a different type.
    ok = [
        'Relu/relu/Nonlinearity',
        'Relu/relu.*/Nonlinearity',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  @test_util.deprecated_graph_mode_only
  def testSingleFunctionElided(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(a):
        return nn.relu(a)

      def body(a):
        return func(a)

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float16, [64, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {a: np.ones(a.shape)})
      self.assertAllClose(result[0], np.broadcast_to(1.0, [64, 64]))

      report_json.parse_log()

      # Function inlined into the entry computation.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 1)

    report = pva.openReport(report_helper.find_report())
    ok = [
        'Relu/relu*/Nonlinearity',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  @test_util.deprecated_graph_mode_only
  def testFunctionTrainingConstants(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(lhs, rhs, a):
        # Number of splits needs to be passed to the grad function.
        rhs_1, rhs_2 = array_ops.split(rhs, 2, -1)
        x1 = math_ops.matmul(lhs, rhs_1)
        x2 = math_ops.matmul(lhs, rhs_2)
        x = array_ops.concat([x1, x2], axis=1)
        x = x + a
        x = math_ops.sigmoid(x)
        return x

      def body(a, b, c, labels):
        with variable_scope.variable_scope("vs", use_resource=True):
          w0 = variable_scope.get_variable(
              "w0",
              shape=[64, 64],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
          w1 = variable_scope.get_variable(
              "w1",
              shape=[64, 64],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
        a = func(a, w0, b)
        a = a - func(a, w1, c)
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=a,
                                                        labels=labels))
        train_op = gradient_descent.GradientDescentOptimizer(0.001).minimize(
            loss)
        return a, train_op

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [64, 64])
        b = array_ops.placeholder(np.float32, [64, 64])
        c = array_ops.placeholder(np.float32, [64, 64])
        labels = array_ops.placeholder(np.int32, [64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b, c, labels])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c, labels]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report_json.parse_log()

      # Entry computastion and 2 outlined ones.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 3)

    report = pva.openReport(report_helper.find_report())
    # There would be multiple non-linearities(grads) if the function was not
    # cached.
    # pylint: disable=line-too-long
    ok = [
        'MatMul/dot*/Conv_1',
        '*/slice-apply*/Op/Add',
        'Sigmoid/sigmoid/Nonlinearity',
        'sub/subtract*/Op/Subtract',
        'Transpose',
        'SparseSoftmaxCrossEntropyWithLogits',
        'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul',
        'gradients/sub_grad/Neg/negate*/Op/Negate',
        'gradients/Sigmoid_grad/SigmoidGrad/sigmoid-grad*/NonLinearityGrad',
        'gradients/AddN/scaled-inplace',
        'gradients/AddN/add*/Op/Add',
        'GradientDescent/update_vs/w*/ResourceApplyGradientDescent/scaled-inplace',
        'gradients/MatMul_grad/MatMul_1/dot',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)
    self.assert_total_tile_memory(report, 290009, tolerance=0.1)
    self.assert_max_tile_memory(report, 37483, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testNoGradient(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(lhs, rhs):
        @custom_gradient.custom_gradient
        def f(a, b):
          def grad(dy):
            return [None, dy - b]

          return a, grad

        return f(lhs, rhs)

      def body(a):
        with variable_scope.variable_scope("vs", use_resource=True):
          w0 = variable_scope.get_variable(
              "w0",
              shape=[64, 64],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
        a = func(a, w0)
        return gradients_impl.gradients(a, [w0])

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [64, 64])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      result = sess.run(res, {x: np.ones(x.shape) for x in [a]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

  @test_util.deprecated_graph_mode_only
  def testInputsWithAliasing(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      @ipu.outlined_function
      def func(x):
        return math_ops.sigmoid(x)

      def body(a, b):
        a = array_ops.broadcast_to(a, shape=[1024])
        b = array_ops.broadcast_to(b, shape=[1024])
        return func(a) - func(b)

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [])
        b = array_ops.placeholder(np.float32, [])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[a, b])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      report_json = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b]})
      self.assertAllClose(result[0], np.broadcast_to(0., [1024]))

      report_json.parse_log()

      # Entry computation and outlined one.
      self.assertEqual(len(report_json.tensor_map.computation_names()), 2)

    report = pva.openReport(report_helper.find_report())
    self.assert_max_tile_memory(report, 2110, tolerance=0.1)

  @test_util.deprecated_graph_mode_only
  def testResourceUpdateErrors(self):
    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()

    def body():
      def resource_update_(accumulation_count):
        ipu.internal_ops.print_tensor(accumulation_count)

      with ops.name_scope("WU") as scope:
        to_print = math_ops.sigmoid(4.0)
        func_graph, captured_args, _ = \
          functional_ops._compile_function(  # pylint: disable=protected-access
              resource_update_, [to_print], scope, [], True)

      # Create the resource update and lower the function into XLA.
      with ops.control_dependencies(list(func_graph.control_captures)):
        outputs = gen_functional_ops.resource_update(
            captured_args,
            to_apply=control_util.create_new_tf_function(func_graph),
            Tout=func_graph.output_types,
            output_shapes=func_graph.output_shapes,
            offload_weight_update_variables=threestate_pb2.ThreeState.Name(
                threestate_pb2.THREESTATE_UNDEFINED),
            replicated_optimizer_state_sharding=threestate_pb2.ThreeState.Name(
                threestate_pb2.THREESTATE_UNDEFINED))
        return outputs

    def my_net():
      return ipu.loops.repeat(5, body, [])

    with ops.device("/device:IPU:0"):
      outputs = ipu.ipu_compiler.compile(my_net, inputs=[])

    with tu.ipu_session() as sess:
      with self.assertRaisesRegex(
          Exception, r"No gradient accumulation count instruction found for "
          r"resource update instruction *"):
        sess.run(variables.global_variables_initializer())
        sess.run(outputs)

  def testPipelineFirstConstant(self):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    input_data = np.array([0., 1., 2., 3.])
    actual_value = np.array([[-123., 0., 123., 246.]]).T

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_ = keras.Input(shape=(1,))

      with ipu.keras.PipelineStage(0):
        x = math_ops.multiply(123., input_)

      with ipu.keras.PipelineStage(1):
        y = math_ops.subtract(x, 123.)

      model = keras.Model(input_, y)
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.compile(optimizer='adam', loss='mse', steps_per_execution=8)

      expected_value = model.predict(input_data, batch_size=2)
      self.assertAllClose(expected_value, actual_value, atol=1e-05)

  def testPipelineFirstLargeConstant(self):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    const_shape = (12, 12)

    input_data = np.array([0., 1., 2., 3.])
    actual_value_numbers = [-123., 0., 123., 246.]
    actual_value = np.zeros(shape=(const_shape[0] * len(input_data),
                                   const_shape[1]))
    # Broadcast the actual_value_numbers to const_shape chunks then stack
    for i, n in enumerate(actual_value_numbers):
      start = i * const_shape[0]
      end = (i + 1) * const_shape[0]
      actual_value[start:end, :] = n

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_ = keras.Input(shape=(1,), dtype=dtypes.float32)

      with ipu.keras.PipelineStage(0):
        x = math_ops.multiply(
            np.full(fill_value=123., shape=const_shape, dtype=np.float32),
            input_)

      with ipu.keras.PipelineStage(1):
        y = math_ops.subtract(x, np.full(fill_value=123., shape=const_shape))

      model = keras.Model(input_, y)
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.compile(optimizer='adam', loss='mse', steps_per_execution=8)

      expected_value = model.predict(input_data, batch_size=1)
      self.assertAllClose(expected_value, actual_value, atol=1e-05)

  def testOutlinedFunctionInFunction(self):
    @ipu.outlined_function
    def identity(x):
      return x

    @def_function.function(experimental_compile=True)
    def f(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        z = identity(x)
      return tape.gradient(z, x)

    config = ipu.config.IPUConfig()
    config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
      x = random_ops.random_normal((1, 10))
      dfdx = strategy.run(f, [x])
      self.assertAllEqual(dfdx, np.ones((1, 10)))

  def testKerasCustomLayerWithOutlinedFunction(self):
    class CustomLayer(keras.layers.Layer):
      def __init__(self, **kwargs):
        self.dense = keras.layers.Dense(units=4)
        super().__init__(**kwargs)

      def build(self, input_shape):
        self.dense.build(input_shape)
        super().build(input_shape)

      def call(self, inputs):  # pylint: disable=arguments-differ
        @ipu.outlined_function
        def inner_call():
          y = self.dense(inputs)
          return y

        return inner_call()

    # Configure the IPU device.
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    micro_batch_size = 4
    ds = dataset_ops.Dataset.from_tensor_slices(
        ([1.] * micro_batch_size * 4, [2.] * micro_batch_size * 4))
    ds = ds.batch(micro_batch_size, drop_remainder=True)

    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
      # Functional model
      input_layer = keras.Input(shape=1, batch_size=micro_batch_size)
      x = CustomLayer()(input_layer)
      model = keras.Model(input_layer, x)

      model.compile(optimizer="sgd",
                    loss=keras.losses.SparseCategoricalCrossentropy())

      model.fit(ds, batch_size=micro_batch_size)


if __name__ == "__main__":
  googletest.main()

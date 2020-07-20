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

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class FunctionalOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testFunctionInferenceWithVariableScope(self):
    with tu.ipu_session() as sess:

      def func(a, b, name):
        @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report.parse_log()
      # There would be multiple non-linearities if the function was not
      # cached.
      ok = [
          'MatMul/dot*/Conv_1',
          'add/add*/Op/Add',
          'Sigmoid/custom-call/Nonlinearity',
          'sub/subtract*/Op/Subtract',
          '__seed',
          'Copy_',
      ]
      report.assert_all_compute_sets_and_list(ok)
      report.assert_total_tile_memory(954492)
      report.assert_max_tile_memory(1690)

      # Entry computation and outlined one.
      self.assertEqual(len(report.tensor_map.computation_names()), 2)

  @test_util.deprecated_graph_mode_only
  def testFunctionTraining(self):
    with tu.ipu_session() as sess:

      @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c, labels]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report.parse_log()
      # There would be multiple non-linearities(grads) if the function was not
      # cached.
      ok = [
          'MatMul/dot*/Conv_1',
          'add/add*/Op/Add',
          'Sigmoid/custom-call/Nonlinearity',
          'sub/subtract*/Op/Subtract',
          '__seed',
          'Copy_',
          'SparseSoftmaxCrossEntropyWithLogits',
          'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul',
          'gradients/sub_grad/Neg/negate*/Op/Negate',
          'gradients/Sigmoid_grad/SigmoidGrad/custom-call*/NonLinearityGrad',
          'gradients/AddN/fusion/scaledAdd/Op/Multiply',
          'gradients/AddN/fusion/AddTo',
          'GradientDescent/update_vs/w*/ResourceApplyGradientDescent/fusion*/AddTo',
          'gradients/AddN/fusion/scaledAdd/Op/Multiply/OnTileCopyPre',
      ]
      report.assert_all_compute_sets_and_list(ok)
      report.assert_total_tile_memory(1193804)
      report.assert_max_tile_memory(4068)

      # Entry computastion and 2 outlined ones.
      self.assertEqual(len(report.tensor_map.computation_names()), 3)

  @test_util.deprecated_graph_mode_only
  def testNestedFunctionTraining(self):
    with tu.ipu_session() as sess:

      def matmul_with_bias(x, scope_name):
        @ipu.function
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
        @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, labels]})
      self.assertAllClose(result[0], np.broadcast_to(1., [64, 64]))

      report.parse_log()
      # There would be multiple non-linearities(grads) if the function was not
      # cached.
      ok = [
          '__seed/set/setMasterSeed',
          'matmul/dot*/Conv_1',
          'add_0/fusion/Op/Add',
          'Sigmoid/custom-call/Nonlinearity',
          'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits',
          'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/',
          'gradients/Sigmoid_grad/SigmoidGrad/custom-call.2/NonLinearityGrad',
          'gradients/add_grad/Sum/reduce*/Reduce',
          'GradientDescent/update_1/bias/ResourceApplyGradientDescent/fusion.5/AddTo',
          'GradientDescent/update_1/w/ResourceApplyGradientDescent/fusion.4/AddTo',
          'GradientDescent/update_2/bias/ResourceApplyGradientDescent/fusion.3/AddTo',
          'GradientDescent/update_2/w/ResourceApplyGradientDescent/fusion.2/AddTo',
          'Copy_',
      ]
      report.assert_all_compute_sets_and_list(ok)
      report.assert_total_tile_memory(1148984)
      report.assert_max_tile_memory(4172)

      # Entry computastion and 4 outlined ones.
      self.assertEqual(len(report.tensor_map.computation_names()), 5)

  @test_util.deprecated_graph_mode_only
  def testFunctionSerializedLookup(self):
    with tu.ipu_session() as sess:

      @ipu.function
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

      report = tu.ReportJSON(self, sess)
      i_h = np.arange(0, DICT_SIZE, step=SPLIT_SIZE // 2)
      w_h = np.arange(EMB_SIZE, dtype=np.float16) * np.ones(
          [DICT_SIZE, EMB_SIZE], dtype=np.float16)
      result = sess.run(res, {table: w_h, indices: i_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))

      report.parse_log()
      # There would be multiple multi slices if the function was not cached.
      ok = [
          'Less/fusion*/Op/LessThan',
          'GreaterEqual/fusion*/Op/GreaterThanEqual',
          'sub/fusion/Op/Subtract',
          'embedding_lookup/custom-call/output/multiSlice',
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
          '__seed',
          'Copy_',
      ]
      report.assert_all_compute_sets_and_list(ok)
      report.assert_total_tile_memory(10980622)
      report.assert_max_tile_memory(9888)

      # Main computation and outlined serialized one.
      self.assertEqual(len(report.tensor_map.computation_names()), 2)

  @test_util.deprecated_graph_mode_only
  def testFunctionsNoMatch(self):
    with tu.ipu_session() as sess:

      @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c]})
      self.assertAllClose(result[0], np.broadcast_to(1.0, [64, 64]))
      self.assertAllClose(result[1], np.broadcast_to(1.0, [64, 64]))
      self.assertAllClose(result[2], np.broadcast_to(1.0, [64, 64]))

      report.parse_log()
      # Two non-linearties, as one of them has a different type.
      ok = [
          'Relu/custom-call/Nonlinearity',
          'Relu/custom-call.*/Nonlinearity',
          '__seed',
          'Copy_',
      ]
      report.assert_all_compute_sets_and_list(ok)

      # Main computation (including inlined fp32 one, and the fp16 outlined).
      self.assertEqual(len(report.tensor_map.computation_names()), 2)

  @test_util.deprecated_graph_mode_only
  def testSingleFunctionElided(self):
    with tu.ipu_session() as sess:

      @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {a: np.ones(a.shape)})
      self.assertAllClose(result[0], np.broadcast_to(1.0, [64, 64]))

      report.parse_log()

      ok = [
          'Relu/custom-call*/Nonlinearity',
          '__seed',
      ]
      report.assert_all_compute_sets_and_list(ok)

      # Function inlined into the entry computation.
      self.assertEqual(len(report.tensor_map.computation_names()), 1)

  @test_util.deprecated_graph_mode_only
  def testFunctionTrainingConstants(self):
    with tu.ipu_session() as sess:

      @ipu.function
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

      report = tu.ReportJSON(self, sess)
      result = sess.run(res, {x: np.ones(x.shape) for x in [a, b, c, labels]})
      self.assertAllClose(result[0], np.broadcast_to(0., [64, 64]))

      report.parse_log()
      # There would be multiple non-linearities(grads) if the function was not
      # cached.
      ok = [
          'MatMul/dot*/Conv_1',
          '*/custom-call*/Op/Add',
          'Sigmoid/custom-call/Nonlinearity',
          'sub/subtract*/Op/Subtract',
          '__seed',
          'Copy_',
          'SparseSoftmaxCrossEntropyWithLogits',
          'gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul',
          'gradients/sub_grad/Neg/negate*/Op/Negate',
          'gradients/Sigmoid_grad/SigmoidGrad/custom-call*/NonLinearityGrad',
          'gradients/AddN/fusion/scaledAdd/Op/Multiply',
          'gradients/AddN/fusion/AddTo',
          'gradients/AddN/add*/Op/Add',
          'GradientDescent/update_vs/w*/ResourceApplyGradientDescent/fusion*/AddTo',
          'gradients/AddN/fusion/scaledAdd/Op/Multiply/OnTileCopyPre',
          'gradients/MatMul_1_grad/MatMul/dot',
      ]
      report.assert_all_compute_sets_and_list(ok)
      report.assert_total_tile_memory(1342820)
      report.assert_max_tile_memory(5186)

      # Entry computastion and 2 outlined ones.
      self.assertEqual(len(report.tensor_map.computation_names()), 3)


if __name__ == "__main__":
  googletest.main()

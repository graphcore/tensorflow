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

import functools
import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest

TEST_CASES = ({
    'testcase_name': 'a_columns',
    'a_shape': [8, 16],
    'b_shape': [16, 5],
    'transpose_a': False,
    'transpose_b': False,
    'serialization_factor': 2,
    'serialization_dimension': 'a_columns',
}, {
    'testcase_name': 'a_columns_tb',
    'a_shape': [32, 8],
    'b_shape': [1, 8],
    'transpose_a': False,
    'transpose_b': True,
    'serialization_factor': 2,
    'serialization_dimension': 'a_columns',
}, {
    'testcase_name': 'a_columns_ta',
    'a_shape': [4, 8],
    'b_shape': [4, 4],
    'transpose_a': True,
    'transpose_b': False,
    'serialization_factor': 2,
    'serialization_dimension': 'a_columns',
}, {
    'testcase_name': 'a_columns_ta_tb',
    'a_shape': [64, 32],
    'b_shape': [128, 64],
    'transpose_a': True,
    'transpose_b': True,
    'serialization_factor': 8,
    'serialization_dimension': 'a_columns',
}, {
    'testcase_name': 'a_rows_b_columns',
    'a_shape': [4, 21],
    'b_shape': [21, 8],
    'transpose_a': False,
    'transpose_b': False,
    'serialization_factor': 3,
    'serialization_dimension': 'a_rows_b_columns',
}, {
    'testcase_name': 'a_rows_b_columns_tb',
    'a_shape': [4, 72],
    'b_shape': [1, 72],
    'transpose_a': False,
    'transpose_b': True,
    'serialization_factor': 2,
    'serialization_dimension': 'a_rows_b_columns',
}, {
    'testcase_name': 'a_rows_b_columns_ta',
    'a_shape': [4, 8],
    'b_shape': [4, 4],
    'transpose_a': True,
    'transpose_b': False,
    'serialization_factor': 2,
    'serialization_dimension': 'a_rows_b_columns',
}, {
    'testcase_name': 'a_rows_b_columns_ta_tb',
    'a_shape': [4, 5],
    'b_shape': [5, 4],
    'transpose_a': True,
    'transpose_b': True,
    'serialization_factor': 4,
    'serialization_dimension': 'a_rows_b_columns',
}, {
    'testcase_name': 'b_rows',
    'a_shape': [4, 4],
    'b_shape': [4, 8],
    'transpose_a': False,
    'transpose_b': False,
    'serialization_factor': 2,
    'serialization_dimension': 'b_rows',
}, {
    'testcase_name': 'b_rows_tb',
    'a_shape': [4, 4],
    'b_shape': [44, 4],
    'transpose_a': False,
    'transpose_b': True,
    'serialization_factor': 2,
    'serialization_dimension': 'b_rows',
}, {
    'testcase_name': 'b_rows_ta',
    'a_shape': [4, 8],
    'b_shape': [4, 4],
    'transpose_a': True,
    'transpose_b': False,
    'serialization_factor': 2,
    'serialization_dimension': 'b_rows',
}, {
    'testcase_name': 'b_rows_ta_tb',
    'a_shape': [4, 5],
    'b_shape': [8, 4],
    'transpose_a': True,
    'transpose_b': True,
    'serialization_factor': 4,
    'serialization_dimension': 'b_rows',
}, {
    'testcase_name': 'a_columns_serial_factor_1',
    'a_shape': [4, 5],
    'b_shape': [5, 4],
    'transpose_a': True,
    'transpose_b': True,
    'serialization_factor': 1,
    'serialization_dimension': 'a_columns',
})


def _getTestCases():
  from copy import deepcopy

  test_cases = list(TEST_CASES)
  # Add test cases with a batch dim for a.
  for case in deepcopy(TEST_CASES):
    case['testcase_name'] += "_batch_a"
    case['a_shape'] = [2] + case['a_shape']
    test_cases.append(case)
  # Add test cases with a batch dim for b.
  for case in deepcopy(TEST_CASES):
    case['testcase_name'] += "_batch_b"
    case['b_shape'] = [3] + case['b_shape']
    test_cases.append(case)
  # Add test cases with a batch dim for a and b.
  for case in deepcopy(TEST_CASES):
    case['testcase_name'] += "_batch_a_batch_b"
    case['a_shape'] = [3] + case['a_shape']
    case['b_shape'] = [3] + case['b_shape']
    test_cases.append(case)
  return test_cases


# Note that in this test we expect small numerical differences as serializing
# means that some operations are done in a different order.
class SerializedMatmulTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(0xDEADBEEF)

  def _testOnCpu(self, model_fn, placeholders, inputs, sess, scope_name=None):
    scope_name = scope_name if scope_name else "cpu_vs"
    with variable_scope.variable_scope(scope_name, use_resource=True):
      output = model_fn(*placeholders)
    sess.run(variables.global_variables_initializer())
    return sess.run(output, inputs)

  def _testOnIpu(self, model_fn, placeholders, inputs, sess, scope_name=None):
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      scope_name = scope_name if scope_name else "ipu_vs"
      with variable_scope.variable_scope(scope_name, use_resource=True):
        output = ipu.ipu_compiler.compile(model_fn, placeholders)
    ipu.utils.move_variable_initialization_to_cpu()
    sess.run(variables.global_variables_initializer())
    return sess.run(output, inputs)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.deprecated_graph_mode_only
  def testSerializedMatmul(self, a_shape, b_shape, transpose_a, transpose_b,
                           serialization_factor, serialization_dimension):
    a = array_ops.placeholder(np.float32, a_shape)
    b = array_ops.placeholder(np.float32, b_shape)

    def cpu_matmul(a, b):
      return math_ops.matmul(a,
                             b,
                             transpose_a=transpose_a,
                             transpose_b=transpose_b)

    def ipu_matmul(a, b):
      return ipu.math_ops.serialized_matmul(a,
                                            b,
                                            serialization_factor,
                                            serialization_dimension,
                                            transpose_a=transpose_a,
                                            transpose_b=transpose_b)

    a_val = np.random.normal(2.0, 2.0, a_shape)
    b_val = np.random.normal(2.0, 2.0, b_shape)

    with sl.Session() as sess:
      cpu_output = self._testOnCpu(cpu_matmul, [a, b], {
          a: a_val,
          b: b_val
      }, sess)
      ipu_output = self._testOnIpu(ipu_matmul, [a, b], {
          a: a_val,
          b: b_val
      }, sess)
      self.assertAllClose(cpu_output, ipu_output[0], atol=1.e-05, rtol=1.e-05)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.deprecated_graph_mode_only
  def testSerializedMatmulGrad(self, a_shape, b_shape, transpose_a,
                               transpose_b, serialization_factor,
                               serialization_dimension):
    a_val = array_ops.constant(np.random.normal(2.0, 2.0, a_shape),
                               dtype=np.float32)
    b_val = array_ops.constant(np.random.normal(2.0, 2.0, b_shape),
                               dtype=np.float32)

    def matmul(a, b):
      return math_ops.matmul(a,
                             b,
                             transpose_a=transpose_a,
                             transpose_b=transpose_b)

    def serialized_matmul(a, b):
      return ipu.math_ops.serialized_matmul(a,
                                            b,
                                            serialization_factor,
                                            serialization_dimension,
                                            transpose_a=transpose_a,
                                            transpose_b=transpose_b)

    def model_fn(matmul_fn):
      a = variable_scope.get_variable("a", initializer=a_val)
      b = variable_scope.get_variable("b", initializer=b_val)
      c = matmul_fn(a, b)
      # Not a real loss function, but good enough for testing backprop.
      loss = math_ops.reduce_sum(c)
      outputs = gradients_impl.gradients(loss, [a, b])
      outputs.append(loss)
      return outputs

    ipu_fn = functools.partial(model_fn, matmul)
    ipu_serial_fn = functools.partial(model_fn, serialized_matmul)

    with sl.Session() as sess:
      a, b, l = self._testOnIpu(ipu_fn, [], {}, sess, "normal")
      serial_a, serial_b, serial_l = self._testOnIpu(ipu_serial_fn, [], {},
                                                     sess, "serial")

      self.assertAllClose(a, serial_a, atol=1.e-05, rtol=1.e-05)
      self.assertAllClose(b, serial_b, atol=1.e-05, rtol=1.e-05)
      self.assertAllClose([l], [serial_l], atol=1.e-05, rtol=1.e-05)


if __name__ == "__main__":
  googletest.main()

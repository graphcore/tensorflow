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

from absl.testing import parameterized

from tensorflow.python.ipu import test_utils as tu
import numpy as np
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.ops.f8_ops import Format, convert_from_f8, convert_to_f8, create_metadata, QuarterTensor, f8_matmul


def _f8_convert_cases():
  cases = []
  float_values = [-0.75, 1, 2, -4]
  int_values = [-1, 1, 2, -4]
  for f in Format:
    for scale in [-1, 0, 1]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.int32):
        for on_ipu in (False, True):
          device = "IPU" if on_ipu else "CPU"
          case = {
              'testcase_name': f"{device}_{f}_{scale}_{dtype.name}",
              'f8_format': f,
              'f8_scale': scale,
              'dtype': dtype,
              'input_values':
              float_values if dtype != dtypes.int32 else int_values,
              'on_ipu': on_ipu
          }
          cases.append(case)
  return cases


F8_CONVERT_CASES = _f8_convert_cases()


class F8Test(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*F8_CONVERT_CASES)
  @test_util.deprecated_graph_mode_only
  def testConvertF8(self, f8_format, f8_scale, dtype, input_values, on_ipu):
    if on_ipu:
      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.configure_ipu_system()

    with ops.device('cpu'):
      a = array_ops.placeholder(dtype, [4])
      m = array_ops.placeholder(dtypes.uint8, [])

    def my_net(values, meta):
      f8 = convert_to_f8(values, meta)
      return convert_from_f8(f8, dtype=dtype)

    if on_ipu:
      with ipu.scopes.ipu_scope('/device:IPU:0'):
        res = ipu.ipu_compiler.compile(my_net, [a, m])
    else:
      with ops.device("/device:CPU:0"):
        res = [my_net(a, m)]

    with tu.ipu_session() as sess:
      result, = sess.run(res, {
          a: input_values,
          m: create_metadata(f8_format, f8_scale)
      })
    self.assertAllEqual(result, input_values)


def convert_to_quarter(x):
  x = tf.cast(x, dtypes.float16)
  x = convert_to_f8(x, create_metadata(Format.F143, 1))
  return QuarterTensor(x[0], x[1])


def cast_to(x, dtype):
  if dtype == dtypes.uint8:
    return convert_to_quarter(x)
  return tf.cast(x, dtype)


def mat_mul_test_cases():
  result = []
  pauli_x_ = [[0, 1], [1, 0]]
  pauli_x = array_ops.constant(pauli_x_)
  pauli_x = tf.cast(pauli_x, dtypes.uint8)

  pauli_z_ = [[2, 0], [0, 2]]
  pauli_z = array_ops.constant(pauli_z_)

  ans = np.matmul(np.array(pauli_x_), np.array(pauli_z_))
  for ltype in [dtypes.uint8]:
    result.append({
        'testcase_name': f"mat_mul_{ltype}",
        'lhs_in': tf.cast(pauli_z, ltype),
        'rhs_in': pauli_x,
        'answer': ans,
    })

  result.append({
      'testcase_name':
      "grouped_mat_mul",
      'lhs_in':
      tf.broadcast_to(tf.cast(pauli_z, dtypes.uint8), [2, 2, 2]),
      'rhs_in':
      tf.broadcast_to(pauli_x, [2, 2, 2]),
      'answer':
      np.broadcast_to(ans, [2, 2, 2]),
  })

  return result


mm_cases = mat_mul_test_cases()


class F8MatMulTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*mm_cases)
  @test_util.deprecated_graph_mode_only
  def testConvertF8(self, lhs_in, rhs_in, answer):
    def model(lhs_, rhs_):
      lhs = cast_to(lhs_, lhs_.dtype)
      rhs = cast_to(rhs_, rhs_.dtype)

      out = f8_matmul(lhs, rhs)
      if isinstance(out, QuarterTensor):
        out = convert_from_f8((out.data, out.metadata), dtype=dtypes.int32)
      else:
        out = tf.cast(out, dtypes.int32)
      return out, rhs_

    with ops.device('cpu'):
      aa = array_ops.placeholder(lhs_in.dtype, lhs_in.shape)
      mm = array_ops.placeholder(rhs_in.dtype, rhs_in.shape)

    with tu.ipu_session() as sess:
      with ipu.scopes.ipu_scope('/device:IPU:0'):
        res = ipu.ipu_compiler.compile(model, [aa, mm])
        result, ans, = sess.run(res, {
            aa: lhs_in.numpy(),
            mm: rhs_in.numpy(),
        })

    self.assertAllEqual(answer, result)


if __name__ == "__main__":
  googletest.main()

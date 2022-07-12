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
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.ops.f8_ops import Format, convert_from_f8, convert_to_f8, create_metadata


def _f8_convert_cases():
  cases = []
  float_values = [-0.75, 1, 2, -4]
  int_values = [-1, 1, 2, -4]
  for f in Format:
    for scale in [-1, 0, 1]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.int32):
        case = {
            'testcase_name': f"{f}_{scale}_{dtype.name}",
            'f8_format': f,
            'f8_scale': scale,
            'dtype': dtype,
            'input_values':
            float_values if dtype != dtypes.int32 else int_values
        }
        cases.append(case)
  return cases


F8_CONVERT_CASES = _f8_convert_cases()


class F8Test(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.named_parameters(*F8_CONVERT_CASES)
  @test_util.deprecated_graph_mode_only
  def testConvertF8(self, f8_format, f8_scale, dtype, input_values):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device('cpu'):
      a = array_ops.placeholder(dtype, [4])
      m = array_ops.placeholder(dtypes.uint8, [])

    def my_net(values, meta):
      f8 = convert_to_f8(values, meta)
      return convert_from_f8(f8, dtype=dtype)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      res = ipu.ipu_compiler.compile(my_net, [a, m])

    with tu.ipu_session() as sess:
      result, = sess.run(res, {
          a: input_values,
          m: create_metadata(f8_format, f8_scale)
      })
    self.assertAllEqual(result, input_values)


if __name__ == "__main__":
  googletest.main()

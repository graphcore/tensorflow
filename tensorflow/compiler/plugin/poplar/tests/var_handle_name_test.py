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
# ==============================================================================

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ipu.dataset_extractor import get_variable_handles
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class VarHandleNameTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def testResourceToVarHandleNameOp(self):
    with self.session() as sess:

      x = resource_variable_ops.ResourceVariable(random_ops.random_normal(
          [5, 5], stddev=0.1),
                                                 name="thisIsTheVariableName")

      sess.run(variables.global_variables_initializer())

      handle_names = [x[0] for x in sess.run(get_variable_handles([x]))]

      self.assertEqual(handle_names, [b'thisIsTheVariableName'])


if __name__ == "__main__":
  googletest.main()

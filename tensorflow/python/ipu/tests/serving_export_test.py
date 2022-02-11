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
# ==============================================================================

import tempfile

import numpy as np

from absl.testing import parameterized
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import serving
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import signature_constants


class TestServingExport(test_util.TensorFlowTestCase, parameterized.TestCase):
  def setUp(self):
    super().setUp()

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

  def _load_and_run(self, path, inputs):
    imported = load.load(path)
    loaded = imported.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    output_name = next(iter(loaded.structured_outputs.values())).name
    input_names = [inp.name.split(':')[0] for inp in loaded.inputs]

    if not isinstance(inputs, dict):
      inputs = {input_names[0]: inputs}

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      result = strategy.run(loaded, kwargs=inputs)

    return result[output_name]

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.run_v2_only
  def test_export_simple_model_no_var(self):

    element_count = 4
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32,
                                              name='x'),)

    @def_function.function
    def my_net(x):
      return x * x

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(my_net, tmp_folder, iterations,
                                 input_signature)

      input_data = np.arange(element_count, dtype=np.float32)

      # load and run
      result = self._load_and_run(tmp_folder, input_data)
      self.assertEqual(list(result), list(np.square(input_data)))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.run_v2_only
  def test_export_simple_model_with_variable(self):
    element_count = 3
    input_shape = (element_count,)
    input_tensor = array_ops.zeros(shape=input_shape, dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensors(input_tensor)

    var_value = np.float16(4.)
    w = variables.Variable(var_value)

    @def_function.function
    def my_net(x):
      return x * w

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(my_net,
                                 tmp_folder,
                                 iterations,
                                 input_dataset=dataset)

      input_data = np.arange(element_count, dtype=np.float16)
      result = self._load_and_run(tmp_folder, input_data)
      self.assertEqual(list(result), list(input_data * var_value))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.run_v2_only
  def test_export_simple_model_two_inputs(self):

    element_count = 3
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32),
                       tensor_spec.TensorSpec(shape=(), dtype=np.float32))

    @def_function.function(input_signature=input_signature)
    def my_net(x1, x2):
      return x1 * x2

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(my_net, tmp_folder, iterations)

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = np.float32(3.0)
      result = self._load_and_run(tmp_folder, {'x1': x1_data, 'x2': x2_data})

      self.assertEqual(list(result), list(x1_data * x2_data))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.run_v2_only
  def test_export_simple_model_run_locally(self):
    element_count = 3
    input_shape = (element_count,)

    inputs = (array_ops.zeros(input_shape, np.float32),
              array_ops.zeros(input_shape, np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)
    var_value = np.float32(4.)
    w = variables.Variable(var_value)

    @def_function.function
    def my_net(x1, x2):
      return (x1 + x2) * w

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      runtime_func = serving.export_single_step(my_net,
                                                tmp_folder,
                                                iterations,
                                                input_dataset=dataset)

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data * 2
      strategy = ipu_strategy.IPUStrategy()
      with strategy.scope():
        x1_data_tf = constant_op.constant(x1_data)
        x2_data_tf = constant_op.constant(x2_data)
        result = strategy.run(runtime_func, args=(x1_data_tf, x2_data_tf))
      result = result[0]
      ref_result = (x1_data + x2_data) * var_value
      self.assertEqual(list(result), list(ref_result))


if __name__ == "__main__":
  test.main()

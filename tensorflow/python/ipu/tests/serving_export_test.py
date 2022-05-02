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

import os
import tempfile

import numpy as np

from absl.testing import parameterized
from tensorflow.python.ipu import test_utils as tu
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ipu import serving
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


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
    signature = loaded.function_def.signature
    input_names = [
        i.name for i in signature.input_arg
        if i.type is not types_pb2.DT_RESOURCE
    ]
    output_name = next(iter(loaded.structured_outputs.values())).name

    g = ops.Graph()
    with g.as_default():
      with session_lib.Session(graph=g) as sess:
        loader.load(sess, [tag_constants.SERVING], path)
        output = g.get_tensor_by_name(output_name)

        feed_dict = {}
        if not isinstance(inputs, dict):
          input_tensor = g.get_tensor_by_name(input_names[0] + ':0')
          feed_dict[input_tensor] = inputs
        else:
          for name in input_names:
            input_tensor = g.get_tensor_by_name(name + ':0')
            feed_dict[input_tensor] = inputs[name]

        result = sess.run(output, feed_dict=feed_dict)
    return result

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_no_var(self):

    element_count = 4
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32,
                                              name='x'),)

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
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_with_variable(self):
    element_count = 3
    input_shape = (element_count,)
    input_tensor = array_ops.zeros(shape=input_shape, dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensors(input_tensor)

    var_value = np.float16(4.)

    def my_net(x):
      w = variables.Variable(var_value)
      return x * w

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(my_net,
                                 tmp_folder,
                                 iterations,
                                 input_dataset=dataset,
                                 variable_initializer=init_variables)

      input_data = np.arange(element_count, dtype=np.float16)
      result = self._load_and_run(tmp_folder, input_data)
      self.assertEqual(list(result), list(input_data * var_value))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
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
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_run_locally(self):
    element_count = 3
    input_shape = (element_count,)

    inputs = (array_ops.zeros(input_shape, np.float32),
              array_ops.zeros(input_shape, np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)
    var_value = np.float32(4.)

    def my_net(x1, x2):
      w = variables.Variable(var_value)
      return (x1 + x2) * w

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      runtime_func = serving.export_single_step(
          my_net,
          tmp_folder,
          iterations,
          input_dataset=dataset,
          variable_initializer=init_variables)

      x1_ph = array_ops.placeholder(dtype=np.float32, shape=input_shape)
      x2_ph = array_ops.placeholder(dtype=np.float32, shape=input_shape)

      result_op = runtime_func(x1_ph, x2_ph)
      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data * 2

      with session_lib.Session() as sess:
        result = sess.run(result_op, {x1_ph: x1_data, x2_ph: x2_data})

      result = result[0]
      ref_result = (x1_data + x2_data) * var_value
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_export_single_step_fails_for_non_empty_dir(self):
    input_signature = (tensor_spec.TensorSpec(shape=(4,),
                                              dtype=np.float32,
                                              name='x'),)

    def my_net(x):
      return x * x

    with tempfile.TemporaryDirectory() as tmp_folder:
      open(os.path.join(tmp_folder, 'dummy_file'), 'w').close()
      with self.assertRaisesRegex(ValueError, "is not empty"):
        serving.export_single_step(my_net, tmp_folder, 16, input_signature)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline(self):

    element_count = 4
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32),
                       tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32))

    var_value = np.float32(4.)

    def stage1(x1, x2):
      w = variables.Variable(var_value)
      return x1 * x2 + w

    def stage2(x):
      return x + 2

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline([stage1, stage2],
                              tmp_folder,
                              iterations=16,
                              device_mapping=[0, 0],
                              input_signature=input_signature,
                              variable_initializer=init_variables)

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data * 2

      result = self._load_and_run(tmp_folder, {'x1': x1_data, 'x2': x2_data})
      ref_result = x1_data * x2_data + var_value + 2
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline_tffunction_signature(self):

    element_count = 4
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=(), dtype=np.float16),
                       tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float16))

    @def_function.function(input_signature=input_signature)
    def stage1(x1, x2):
      return x1 * x2

    def stage2(x):
      return x + 2

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline([stage1, stage2],
                              tmp_folder,
                              iterations=16,
                              inputs=[np.float16(42.0)],
                              device_mapping=[0, 0])

      x2_data = np.arange(element_count, dtype=np.float16)
      result = self._load_and_run(tmp_folder, x2_data)
      ref_result = 42.0 * x2_data + 2
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline_dataset_signature(self):

    element_count = 4
    input_shape = (element_count,)

    inputs = (array_ops.zeros(shape=input_shape, dtype=np.float32),
              array_ops.zeros(shape=(), dtype=np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)

    def stage1(x1, x2, x3):
      return x1 * x2 + x3

    def stage2(x):
      return x + 2

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline([stage1, stage2],
                              tmp_folder,
                              iterations=16,
                              inputs=[42.0],
                              device_mapping=[0, 0],
                              input_dataset=dataset)

      x2_data = np.arange(element_count, dtype=np.float32)
      x3_data = np.float32(5.0)
      result = self._load_and_run(tmp_folder, {'x2': x2_data, 'x3': x3_data})
      ref_result = 42.0 * x2_data + x3_data + 2
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline_fails_for_non_empty_dir(self):
    input_signature = (tensor_spec.TensorSpec(shape=(4,), dtype=np.float32))

    def stage(x):
      return x + 2

    with tempfile.TemporaryDirectory() as tmp_folder:
      open(os.path.join(tmp_folder, 'dummy_file'), 'w').close()
      with self.assertRaisesRegex(ValueError, "is not empty"):
        serving.export_pipeline([stage, stage],
                                tmp_folder,
                                iterations=16,
                                device_mapping=[0, 0],
                                input_signature=input_signature)


if __name__ == "__main__":
  test.main()

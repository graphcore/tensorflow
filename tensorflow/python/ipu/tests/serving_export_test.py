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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ipu import serving
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
import popef


class TestServingExport(test_util.TensorFlowTestCase, parameterized.TestCase):
  def setUp(self):
    super().setUp()

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

  def _load_and_run(self, path, inputs, output_names=None):
    imported = load.load(path)
    loaded = imported.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    signature = loaded.function_def.signature
    input_names = [
        i.name for i in signature.input_arg
        if i.type is not types_pb2.DT_RESOURCE
    ]

    if output_names is None and len(loaded.outputs) > 1:
      raise ValueError(
          "_load_and_run `output_names` not set. To preserve "
          "proper order of the outputs, the field `output_names` must be set in "
          "the case when exported model has more than 1 output.")

    if output_names is None or not output_names:
      output_names = list(loaded.structured_outputs.keys())

    # Get tensors' names for given output names.
    tensors_names = [loaded.structured_outputs[n].name for n in output_names]

    g = ops.Graph()
    with g.as_default():
      with session_lib.Session(graph=g) as sess:
        loader.load(sess, [tag_constants.SERVING], path)
        output_tensors = [g.get_tensor_by_name(n) for n in tensors_names]

        feed_dict = {}
        if not isinstance(inputs, dict):
          input_tensor = g.get_tensor_by_name(input_names[0] + ':0')
          feed_dict[input_tensor] = tensor_util.constant_value(inputs)
        else:
          for name in input_names:
            input_tensor = g.get_tensor_by_name(name + ':0')
            feed_dict[input_tensor] = inputs[name]

        results = sess.run(output_tensors, feed_dict=feed_dict)

    return results[0] if len(results) == 1 else results

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_programs_names(self):
    element_count = 3
    input_shape = (element_count,)
    input_signatures = (tensor_spec.TensorSpec(shape=input_shape,
                                               dtype=np.float16,
                                               name='in_x'),
                        tensor_spec.TensorSpec(shape=input_shape,
                                               dtype=np.float16,
                                               name='in_y'))
    var_value = np.float16(4.)

    def model_fn(x, y):
      w = variables.Variable(var_value)
      return x * y + w

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 input_signatures,
                                 variable_initializer=init_variables,
                                 output_names="out_z")

      assets_dir = os.path.join(tmp_folder, 'assets')
      popef_dir = os.path.join(assets_dir, os.listdir(assets_dir)[0])
      reader = popef.Reader()
      reader.parseFile(popef_dir)
      metadata = reader.metadata()[0]
      anchors = metadata.anchors()

      main_prog_idx = 1
      for idx in range(3):
        self.assertEqual(anchors[idx].programs(), [
            main_prog_idx,
        ])

      programs_map = metadata.programsMap()
      self.assertEqual(programs_map[0], 'load_program')
      self.assertEqual(programs_map[1], 'main_program')
      self.assertEqual(programs_map[2], 'save_program')

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_no_var_pre_post_processing_transform_strings(
      self):
    element_count = 2
    input_shape = (element_count,)

    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32,
                                                     name='x'),)

    input_sig = (tensor_spec.TensorSpec(shape=input_shape,
                                        dtype=dtypes.string,
                                        name="input string"),)

    @def_function.function(input_signature=input_sig)
    def preprocessing_step(input_tensor):
      def transform_fn(inp):
        is_gc = lambda: constant_op.constant(1.0)
        is_oth = lambda: constant_op.constant(2.0)
        condition = math_ops.equal(
            inp, constant_op.constant("graphcore", dtype=dtypes.string))
        return control_flow_ops.cond(condition, is_gc, is_oth)

      return array_ops.stack(
          [transform_fn(elem) for elem in array_ops.unstack(input_tensor)])

    @def_function.function(input_signature=(tensor_spec.TensorSpec(
        shape=(2,), dtype=dtypes.float32, name="input"),))
    def postprocessing_step(x):
      return x - 1.0

    def model_fn(x):
      return x * x

    input_data = ["graphcore", "other"]
    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 4
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 predict_step_signature,
                                 preprocessing_step=preprocessing_step,
                                 postprocessing_step=postprocessing_step,
                                 output_names="out0")

      result = self._load_and_run(tmp_folder, input_data, ["out0"])
      self.assertEqual(list(result), [0.0, 3.0])

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_no_var(self):

    element_count = 4
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32,
                                                     name='x'),
                              tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32,
                                                     name='y'))
    preprocessing_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                      dtype=np.float32,
                                                      name='x'),)

    def model_fn(x, y):
      return x * y

    def preprocessing(x):
      return x, x * 10

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(
          model_fn,
          tmp_folder,
          iterations,
          predict_step_signature=predict_step_signature,
          preprocessing_step=preprocessing,
          preprocessing_step_signature=preprocessing_signature,
          output_names="result")

      input_data = np.arange(element_count, dtype=np.float32)

      # load and run
      result = self._load_and_run(tmp_folder, input_data, ["result"])
      self.assertEqual(list(result), list(input_data * input_data * 10))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_no_var_preprocessing(self):

    element_count = 15
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32,
                                                     name='x'),)

    def preprocessing(x):
      return x - 1.0

    def model_fn(x):
      return x * x

    def expected_value(x):
      return list(np.square(np.subtract(x, 1.0)))

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 4
      serving.export_single_step(
          model_fn,
          tmp_folder,
          iterations,
          predict_step_signature,
          preprocessing_step=preprocessing,
          preprocessing_step_signature=predict_step_signature,
          output_names="out0")

      input_data = np.arange(element_count, dtype=np.float32)

      # load and run
      result = self._load_and_run(tmp_folder, input_data)
      self.assertEqual(list(result), expected_value(input_data))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_no_var_postprocessing(self):

    element_count = 15
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32,
                                                     name='x'),)

    postprocessing_step_signature = predict_step_signature

    @def_function.function
    def my_net(x):
      return x * x

    @def_function.function
    def postprocessing_step(x):
      return x - 1.0

    def expected_value(x):
      return list(np.subtract(np.square(x), 1.0))

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 4
      serving.export_single_step(
          my_net,
          tmp_folder,
          iterations,
          predict_step_signature,
          postprocessing_step=postprocessing_step,
          postprocessing_step_signature=postprocessing_step_signature,
          output_names="out0")

      input_data = np.arange(element_count, dtype=np.float32)

      # load and run
      result = self._load_and_run(tmp_folder, input_data)
      self.assertEqual(list(result), expected_value(input_data))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_two_inputs_pre_post_processing(self):
    element_count = 3
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32),
                              tensor_spec.TensorSpec(shape=(),
                                                     dtype=np.float32))

    postprocessing_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float32),
                                     tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float32))

    @def_function.function(input_signature=predict_step_signature)
    def preprocessing(x1, x2):
      return x1 - 2, x2 + 5

    @def_function.function(input_signature=predict_step_signature)
    def model_fn(x1, x2):
      return x1 * x2, x1 + x2

    @def_function.function(input_signature=postprocessing_step_signature)
    def postprocessing_step(x1, x2):
      return math_ops.reduce_sum(x1), math_ops.abs(x2)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 preprocessing_step=preprocessing,
                                 postprocessing_step=postprocessing_step,
                                 output_names=["result0", "result1"])

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = np.float32(3.0)
      result0, result1 = self._load_and_run(tmp_folder, {
          'x1': x1_data,
          'x2': x2_data
      }, ["result0", "result1"])
      x1_data -= 2
      x2_data += 5

      self.assertEqual(float(result0), np.sum(x1_data * x2_data))
      self.assertEqual(list(result1), list(np.absolute(x1_data + x2_data)))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_with_variable(self):
    element_count = 3
    input_shape = (element_count,)
    input_tensor = array_ops.zeros(shape=input_shape, dtype=np.float16)
    dataset = dataset_ops.Dataset.from_tensors(input_tensor)

    var_value = np.float16(4.)

    def model_fn(x):
      w = variables.Variable(var_value)
      return x * w

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 input_dataset=dataset,
                                 variable_initializer=init_variables,
                                 output_names="output_0")

      input_data = np.arange(element_count, dtype=np.float16)
      result = self._load_and_run(tmp_folder, input_data, ["output_0"])
      self.assertEqual(list(result), list(input_data * var_value))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_with_variable_pre_post_processing(self):
    element_count = 17
    input_shape = (element_count,)
    input_tensor = array_ops.zeros(shape=input_shape, dtype=np.float16)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float16),)

    dataset = dataset_ops.Dataset.from_tensors(input_tensor)

    preproc_value = np.float16(2.)
    var_value = np.float16(4.)

    def preprocessing_step(x):
      p = variables.Variable(preproc_value)
      return x * p

    def model_fn(x):
      w = variables.Variable(var_value)
      return x * w

    def postprocessing_step(x):
      p = variables.Variable(preproc_value)
      return x / p

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(
          model_fn,
          tmp_folder,
          iterations,
          predict_step_signature=predict_step_signature,
          input_dataset=dataset,
          variable_initializer=init_variables,
          preprocessing_step=preprocessing_step,
          postprocessing_step_signature=predict_step_signature,
          postprocessing_step=postprocessing_step)

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
    def model_fn(x1, x2):
      return x1 * x2, x1 + x2

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 output_names=["result0", "result1"])

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = np.float32(3.0)
      result0, result1 = self._load_and_run(tmp_folder, {
          'x1': x1_data,
          'x2': x2_data
      }, ["result0", "result1"])

      self.assertEqual(list(result0), list(x1_data * x2_data))
      self.assertEqual(list(result1), list(x1_data + x2_data))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_run_locally(self):
    element_count = 3
    input_shape = (element_count,)

    inputs = (array_ops.zeros(input_shape, np.float32),
              array_ops.zeros(input_shape, np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)
    var_value = np.float32(4.)

    def model_fn(x1, x2):
      w = variables.Variable(var_value)
      return (x1 + x2) * w

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      runtime_func = serving.export_single_step(
          model_fn,
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

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_run_post_pre_post_processing_locally(self):
    element_count = 3
    input_shape = (element_count,)

    inputs = (array_ops.zeros(input_shape, np.float32),
              array_ops.zeros(input_shape, np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32),
                              tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32))
    postprocessing_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                       dtype=np.float32),)
    var_value = np.float32(4.)

    @def_function.function
    def preprocessing_step(x1, x2):
      return x1 - 2, x2 + 5

    def model_fn(x1, x2):
      w = variables.Variable(var_value)
      return (x1 + x2) * w

    @def_function.function(input_signature=postprocessing_signature)
    def postprocessing_step(x):
      return x * 10

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      runtime_func = serving.export_single_step(
          model_fn,
          tmp_folder,
          iterations,
          input_dataset=dataset,
          predict_step_signature=predict_step_signature,
          preprocessing_step=preprocessing_step,
          postprocessing_step=postprocessing_step,
          variable_initializer=init_variables)

      x1_ph = array_ops.placeholder(dtype=np.float32, shape=input_shape)
      x2_ph = array_ops.placeholder(dtype=np.float32, shape=input_shape)

      result_op = runtime_func(x1_ph, x2_ph)
      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data * 2

      with session_lib.Session() as sess:
        result = sess.run(result_op, {x1_ph: x1_data, x2_ph: x2_data})

      result = result[0]
      ref_result = ((x1_data - 2) + (x2_data + 5)) * var_value * 10
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=True)
  @test_util.deprecated_graph_mode_only
  def test_export_single_step_fails_for_non_empty_dir(self):
    input_signature = (tensor_spec.TensorSpec(shape=(4,),
                                              dtype=np.float32,
                                              name='x'),)

    def model_fn(x):
      return x * x

    with tempfile.TemporaryDirectory() as tmp_folder:
      open(os.path.join(tmp_folder, 'dummy_file'), 'w').close()
      with self.assertRaisesRegex(ValueError, "is not empty"):
        serving.export_single_step(model_fn, tmp_folder, 16, input_signature)

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
      return x + 2, x * 3

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline([stage1, stage2],
                              tmp_folder,
                              iterations=16,
                              device_mapping=[0, 0],
                              predict_step_signature=input_signature,
                              variable_initializer=init_variables,
                              output_names=["out0", "out1"])

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data

      out0, out1 = self._load_and_run(tmp_folder, {
          'x1': x1_data,
          'x2': x2_data
      }, ["out0", "out1"])
      ref_result0 = x1_data * x2_data + var_value + 2
      ref_result1 = (x1_data * x2_data + var_value) * 3
      self.assertEqual(list(out0), list(ref_result0))
      self.assertEqual(list(out1), list(ref_result1))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline_pre_post_processing(self):

    element_count = 4
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32),
                              tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32))
    postprocessing_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float32),)
    var_value = np.float32(4.)

    def preprocessing_step(x1, x2):
      return x1 * 10, x2 * x2

    def stage1(x1, x2):
      w = variables.Variable(var_value)
      return x1 * x2 + w

    def stage2(x):
      w = variables.Variable(var_value)
      return x - 2 * w + 2

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    def postprocessing_step(x):
      return math_ops.reduce_sum(x)

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline(
          [stage1, stage2],
          tmp_folder,
          iterations=16,
          device_mapping=[0, 0],
          variable_initializer=init_variables,
          predict_step_signature=predict_step_signature,
          preprocessing_step_signature=predict_step_signature,
          preprocessing_step=preprocessing_step,
          postprocessing_step=postprocessing_step,
          postprocessing_step_signature=postprocessing_step_signature)

      x1_data = np.arange(element_count, dtype=np.float32)
      x2_data = x1_data * 2

      result = self._load_and_run(tmp_folder, {'x1': x1_data, 'x2': x2_data})
      ref_result = np.sum((x1_data * 10) * np.square(x2_data) - var_value + 2)
      self.assertEqual(float(result), float(ref_result))

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
  def test_export_pipeline_tffunction_pre_post_processing_signature(self):
    element_count = 4
    input_shape = (element_count,)
    predict_step_signature = (tensor_spec.TensorSpec(shape=(),
                                                     dtype=np.float16),
                              tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float16))
    preprocessing_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                           dtype=np.float16),)

    postprocessing_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float16),
                                     tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float16))
    output_names = ["out0", "out1"]

    @def_function.function(input_signature=preprocessing_step_signature)
    def preprocessing_step(x2):
      return x2 * x2

    @def_function.function(input_signature=predict_step_signature)
    def stage1(x1, x2):
      return x1 * x2

    def stage2(x):
      return x + 2, x * 3

    @def_function.function(input_signature=postprocessing_step_signature)
    def postprocessing_step(x1, x2):
      return math_ops.cast(x1,
                           dtypes.float32), math_ops.cast(x2, dtypes.float32)

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline([stage1, stage2],
                              tmp_folder,
                              iterations=16,
                              inputs=[np.float16(42.0)],
                              predict_step_signature=predict_step_signature,
                              preprocessing_step=preprocessing_step,
                              postprocessing_step=postprocessing_step,
                              device_mapping=[0, 0],
                              output_names=output_names)

      x2_data = np.arange(element_count, dtype=np.float16)
      ref_out0 = 42.0 * np.square(x2_data) + 2
      ref_out1 = 42.0 * np.square(x2_data) * 3

      result0, result1 = self._load_and_run(tmp_folder, x2_data, output_names)
      self.assertEqual(list(result0), list(ref_out0.astype(np.float32)))
      self.assertEqual(list(result1), list(ref_out1.astype(np.float32)))

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
                                predict_step_signature=input_signature)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_pipeline_dataset_pre_post_processing_signature(self):

    element_count = 4
    input_shape = (element_count,)

    inputs = (array_ops.zeros(shape=input_shape, dtype=np.float32),
              array_ops.zeros(shape=(), dtype=np.float32))
    dataset = dataset_ops.Dataset.from_tensors(inputs)
    predict_step_signature = (tensor_spec.TensorSpec(shape=(),
                                                     dtype=np.float32),
                              tensor_spec.TensorSpec(shape=input_shape,
                                                     dtype=np.float32),
                              tensor_spec.TensorSpec(shape=(),
                                                     dtype=np.float32))
    postprocessing_step_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                                            dtype=np.float32),)

    def preprocessing_step(x2, x3):
      return x2 * x3, x3 + 5

    @def_function.function(input_signature=predict_step_signature)
    def stage1(x1, x2, x3):
      return x1 * x2 + x3

    def stage2(x):
      return x + 2

    def postprocessing_step(x):
      return x - 20

    with tempfile.TemporaryDirectory() as tmp_folder:
      serving.export_pipeline(
          [stage1, stage2],
          tmp_folder,
          iterations=16,
          inputs=[42.0],
          device_mapping=[0, 0],
          input_dataset=dataset,
          preprocessing_step=preprocessing_step,
          postprocessing_step=postprocessing_step,
          postprocessing_step_signature=postprocessing_step_signature)

      x2_data = np.arange(element_count, dtype=np.float32)
      x3_data = np.float32(5.0)
      ref_result = (42.0 * (x2_data * x3_data) + (x3_data + 5) + 2) - 20
      result = self._load_and_run(tmp_folder, {'x2': x2_data, 'x3': x3_data})
      self.assertEqual(list(result), list(ref_result))

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_lstm(self):
    # Regression test to make sure that all variables are getting frozen in
    # the case when Identity op follows ReadVariableOp, what is the case in
    # LSTM Cell.
    input_shape = (1, 1, 1)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32,
                                              name='x'),)
    output_names = ["output", "c_state", "h_state"]

    def model_fn(x):
      lstm_cell = rnn_cell.LSTMCell(num_units=1)
      out = rnn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=dtypes.float32)
      return out

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      iterations = 16
      serving.export_single_step(model_fn,
                                 tmp_folder,
                                 iterations,
                                 predict_step_signature=input_signature,
                                 variable_initializer=init_variables,
                                 output_names=output_names)

      input_data = np.ones(shape=input_shape, dtype=np.float32)
      # Make sure that all variables were frozen and model can be executed
      # with just a real input data.
      self._load_and_run(tmp_folder, input_data, output_names=output_names)

  @tu.test_uses_ipus(num_ipus=1, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def test_export_simple_model_restore_checkpoint(self):
    element_count = 3
    input_shape = (element_count,)
    input_signature = (tensor_spec.TensorSpec(shape=input_shape,
                                              dtype=np.float32,
                                              name='x'),)

    model_fn_multiplier = np.float32(2.0)
    initial_variable_value = np.float32(1.0)
    updated_variable_value = np.float32(5.0)

    def preprocessing(x):
      v = variables.Variable(initial_variable_value)
      return x * v

    def model_fn(x):
      return x * model_fn_multiplier

    def init_variables(sess):
      init = variables.global_variables_initializer()
      sess.run(init)

    with tempfile.TemporaryDirectory() as tmp_folder:
      saved_model_dir = os.path.join(tmp_folder, 'saved_model_dir')
      chkpnt_dir = os.path.join(tmp_folder, 'chkpnt_dir')
      chkpnt_full_path = os.path.join(chkpnt_dir, 'chkpnt')

      g = ops.Graph()
      with g.as_default(), session_lib.Session(graph=g).as_default() as sess:
        input_placeholder = array_ops.placeholder(dtype=np.float32,
                                                  shape=input_shape)
        preprocessing(input_placeholder)
        init_variables(sess)

        # Assign new value to the preprocessing's variable
        var_list = g.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES)
        new_v = constant_op.constant(updated_variable_value)
        v_ = state_ops.assign(var_list[0], new_v)
        sess.run(v_)

        # Save checkpoint for further restore done in export_single_step
        saver_op = saver.Saver()
        saver_op.save(sess, chkpnt_full_path)

      iterations = 16
      output_name = "output_0"
      serving.export_single_step(model_fn,
                                 saved_model_dir,
                                 iterations,
                                 predict_step_signature=input_signature,
                                 variable_initializer=init_variables,
                                 output_names=output_name,
                                 preprocessing_step=preprocessing,
                                 preprocessing_step_signature=input_signature,
                                 checkpoint_restore_dir=chkpnt_dir)

      input_data = np.arange(element_count, dtype=np.float16) + 1
      result = self._load_and_run(saved_model_dir, input_data, [output_name])
      self.assertEqual(
          list(result),
          list(input_data * model_fn_multiplier * updated_variable_value))


if __name__ == "__main__":
  test.main()

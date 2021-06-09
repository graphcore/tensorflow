# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
import tensorflow.compiler.plugin.poplar.tests.test_utils as tu
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import googletest


def _getFrontendAttributes(op):
  try:
    attributes = xla_data_pb2.FrontendAttributes()
    attributes.ParseFromString(op.get_attr(
        ipu.scopes.FRONTEND_ATTRIBUTES_NAME))
    return attributes
  except ValueError:
    return None


def _createInputs(dimensions, dtype):
  pa = array_ops.placeholder(dtype, dimensions)
  pb = array_ops.placeholder(dtype, dimensions)
  return (pa, pb, _createFeeders([pa, pb], dimensions, dtype))


def _createFeeders(inputs, dimensions, dtype):
  return {input: np.zeros(dimensions, dtype=dtype) for input in inputs}


class FrontendAttributesTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testSimpleSingleAttribute(self):
    with ops.device("/device:IPU:0"):
      op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop1")
      with ipu.scopes.frontend_attribute("attr_a", "a"):
        op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop2")
        attributes2 = _getFrontendAttributes(op2)
        self.assertIsNone(_getFrontendAttributes(op1))
        self.assertEqual(attributes2.map.get("attr_a"), "a")

  @test_util.deprecated_graph_mode_only
  def testSimpleMultipleAttributes(self):
    with ops.device("/device:IPU:0"):
      op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop1")
      with ipu.scopes.frontend_attribute("attr_a", "a"):
        op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop2")
        with ipu.scopes.frontend_attribute("attr_b", "b"):
          op3 = ops.get_default_graph().create_op("FloatOutput", [],
                                                  [dtypes.float32],
                                                  name="myop3")
          attributes2 = _getFrontendAttributes(op2)
          attributes3 = _getFrontendAttributes(op3)
          self.assertIsNone(_getFrontendAttributes(op1))
          self.assertEqual(attributes2.map.get("attr_a"), "a")
          self.assertIsNone(attributes2.map.get("attr_b"))
          self.assertEqual(attributes3.map.get("attr_a"), "a")
          self.assertEqual(attributes3.map.get("attr_b"), "b")

  @test_util.deprecated_graph_mode_only
  def testSingleAttributeWithScopes(self):
    op1 = None
    op2 = None
    op3 = None
    op4 = None
    with ops.device("/device:IPU:0"):
      with ipu.scopes.frontend_attribute("attr_a", "a"):
        op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop1")
        with ipu.scopes.frontend_attribute("attr_a", "c"):
          op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                  [dtypes.float32],
                                                  name="myop2")
        op3 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop3")
      op4 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop4")
      attributes1 = _getFrontendAttributes(op1)
      attributes2 = _getFrontendAttributes(op2)
      attributes3 = _getFrontendAttributes(op3)
      self.assertEqual(attributes1.map.get("attr_a"), "a")
      self.assertEqual(attributes2.map.get("attr_a"), "c")
      self.assertEqual(attributes3.map.get("attr_a"), "a")
      self.assertIsNone(_getFrontendAttributes(op4))

  @test_util.deprecated_graph_mode_only
  def testMultipleAttributesWithScopes(self):
    op1 = None
    op2 = None
    op3 = None
    op4 = None
    with ops.device("/device:IPU:0"):
      with ipu.scopes.frontend_attribute("attr_a", "a"):
        op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop1")
        with ipu.scopes.frontend_attribute("attr_a", "c"):
          with ipu.scopes.frontend_attribute("attr_b", "b"):
            op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                    [dtypes.float32],
                                                    name="myop2")
        op3 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop3")
      op4 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop4")
      attributes1 = _getFrontendAttributes(op1)
      attributes2 = _getFrontendAttributes(op2)
      attributes3 = _getFrontendAttributes(op3)
      self.assertEqual(attributes1.map.get("attr_a"), "a")
      self.assertIsNone(attributes1.map.get("attr_b"))
      self.assertEqual(attributes2.map.get("attr_a"), "c")
      self.assertEqual(attributes2.map.get("attr_b"), "b")
      self.assertEqual(attributes3.map.get("attr_a"), "a")
      self.assertIsNone(attributes3.map.get("attr_b"))
      self.assertIsNone(_getFrontendAttributes(op4))

  @test_util.deprecated_graph_mode_only
  def testStochasticRounding(self):
    op1 = None
    op2 = None
    op3 = None
    op4 = None
    op5 = None
    with ops.device("/device:IPU:0"):
      with ipu.scopes.stochastic_rounding(False):
        with ipu.scopes.stochastic_rounding(True):
          op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                                  [dtypes.float32],
                                                  name="myop1")
          with ipu.scopes.stochastic_rounding(False):
            with ipu.scopes.frontend_attribute("attr_b", "b"):
              op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                      [dtypes.float32],
                                                      name="myop2")
          op3 = ops.get_default_graph().create_op("FloatOutput", [],
                                                  [dtypes.float32],
                                                  name="myop3")
        op4 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop4")
      op5 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop5")
      attributes1 = _getFrontendAttributes(op1)
      attributes2 = _getFrontendAttributes(op2)
      attributes3 = _getFrontendAttributes(op3)
      attributes4 = _getFrontendAttributes(op4)
      attributes5 = _getFrontendAttributes(op5)
      self.assertEqual(
          attributes1.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON))
      self.assertIsNone(attributes1.map.get("attr_b"))
      self.assertEqual(
          attributes2.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF))
      self.assertEqual(attributes2.map.get("attr_b"), "b")
      self.assertEqual(
          attributes3.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON))
      self.assertIsNone(attributes3.map.get("attr_b"))
      self.assertEqual(
          attributes4.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF))

      self.assertIsNone(attributes4.map.get("attr_b"))
      self.assertEqual(
          attributes5.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED))
      self.assertIsNone(attributes5.map.get("attr_b"))

  @test_util.deprecated_graph_mode_only
  def testMatMulPartialsType(self):
    with self.session() as sess:
      outputs = {}
      with ops.device("/device:IPU:0"):
        with ipu.scopes.partials_type(np.float32):
          pa, pb, fd = _createInputs([64, 64], np.float16)
          output = math_ops.matmul(pa, pb)
          outputs[output] = ("poplin::ConvPartial*<half,float", fd)
        with ipu.scopes.partials_type(np.float16):
          pa, pb, fd = _createInputs([3, 3], np.float16)
          output = math_ops.matmul(pa, pb)
          outputs[output] = ("poplin::ConvPartial*<half,half", fd)
          with ipu.scopes.partials_type(np.float32):
            pa, pb, fd = _createInputs([32, 32], np.float16)
            output = math_ops.matmul(pa, pb)
            outputs[output] = ("poplin::ConvPartial*<half*float", fd)
          pa, pb, fd = _createInputs([5, 5], np.float16)
          output = math_ops.matmul(pa, pb)
          outputs[output] = ("poplin::ConvPartial*<half,half", fd)

      report = tu.ReportJSON(self, sess)

      for output, expected_output in outputs.items():
        report.reset()

        sess.run(output, expected_output[1])

        report.parse_log()
        report.assert_vertices_contain_list([expected_output[0]])

  @test_util.deprecated_graph_mode_only
  def testLSTMPartialsType(self):
    ops.reset_default_graph()
    with self.session() as sess:
      dtype = np.float16
      batch_size = 1
      seq_len = 32
      input_size = 32
      num_channels = 8
      forget_bias = 0.
      weights_value = 1.
      outputs = []
      with ops.device("/device:IPU:0"):

        def createLSTM(expected_output):
          pinputs = array_ops.placeholder(dtype,
                                          [seq_len, batch_size, input_size],
                                          name="inputs")
          pinitial_h_state = array_ops.placeholder(dtype,
                                                   [batch_size, num_channels],
                                                   name="init_h_state")
          pinitial_c_state = array_ops.placeholder(dtype,
                                                   [batch_size, num_channels],
                                                   name="init_c_state")

          def createLSTMCell(pinputs, pinitial_h_state, pinitial_c_state):
            lstm_cell = rnn_cell.LSTMCell(
                num_channels,
                name='basic_lstm_cell',
                forget_bias=forget_bias,
                initializer=init_ops.constant_initializer(weights_value,
                                                          dtype=dtype),
                reuse=variable_scope.AUTO_REUSE)
            state = rnn_cell.LSTMStateTuple(pinitial_c_state, pinitial_h_state)
            outputs, _ = rnn.dynamic_rnn(lstm_cell,
                                         pinputs,
                                         dtype=dtype,
                                         initial_state=state,
                                         time_major=True)
            return outputs

          r = ipu.ipu_compiler.compile(
              createLSTMCell,
              inputs=[pinputs, pinitial_h_state, pinitial_c_state])

          inputs = np.zeros([seq_len, batch_size, input_size], dtype=dtype)
          initial_h_state = np.zeros([batch_size, num_channels], dtype=dtype)
          initial_c_state = np.zeros([batch_size, num_channels], dtype=dtype)
          fd = {
              pinputs: inputs,
              pinitial_h_state: initial_h_state,
              pinitial_c_state: initial_c_state,
          }
          return (r, expected_output, fd)

        with ipu.scopes.partials_type(np.float16):
          outputs.append(createLSTM("poplin::ConvPartial*<half,half"))
          with ipu.scopes.partials_type(np.float32):
            outputs.append(createLSTM("poplin::ConvPartial*<half,float"))
          outputs.append(createLSTM("poplin::ConvPartial*<half,half"))

      report = tu.ReportJSON(self, sess)

      for output, expected_output, fd in outputs:
        sess.run(gen_ipu_ops.ipu_clear_all_xla_compilation_caches())
        sess.run(variables.global_variables_initializer())

        report.reset()
        sess.run(output, fd)
        report.parse_log()
        report.assert_vertices_contain_list([expected_output])

  @test_util.deprecated_graph_mode_only
  def testUnsupportedPartialsType(self):
    with self.assertRaisesRegex(
        ValueError, "Only support float16, float32, provided float64"):
      with ipu.scopes.partials_type(np.float64):
        pass


if __name__ == "__main__":
  googletest.main()

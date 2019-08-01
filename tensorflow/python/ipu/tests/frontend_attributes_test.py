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

from tensorflow.python.platform import googletest

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python import ipu
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2


def get_frontend_attributes(op):
  try:
    attributes = xla_data_pb2.FrontendAttributes()
    attributes.ParseFromString(op.get_attr(
        ipu.scopes.FRONTEND_ATTRIBUTES_NAME))
    return attributes
  except ValueError:
    return None


class FrontendAttributesTest(test_util.TensorFlowTestCase):
  def testSimpleSingleAttribute(self):
    with ops.device("/device:IPU:0"):
      op1 = ops.get_default_graph().create_op("FloatOutput", [],
                                              [dtypes.float32],
                                              name="myop1")
      with ipu.scopes.frontend_attribute("attr_a", "a"):
        op2 = ops.get_default_graph().create_op("FloatOutput", [],
                                                [dtypes.float32],
                                                name="myop2")
        attributes2 = get_frontend_attributes(op2)
        self.assertIsNone(get_frontend_attributes(op1))
        self.assertEqual(attributes2.map.get("attr_a"), "a")

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
          attributes2 = get_frontend_attributes(op2)
          attributes3 = get_frontend_attributes(op3)
          self.assertIsNone(get_frontend_attributes(op1))
          self.assertEqual(attributes2.map.get("attr_a"), "a")
          self.assertIsNone(attributes2.map.get("attr_b"))
          self.assertEqual(attributes3.map.get("attr_a"), "a")
          self.assertEqual(attributes3.map.get("attr_b"), "b")

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
      attributes1 = get_frontend_attributes(op1)
      attributes2 = get_frontend_attributes(op2)
      attributes3 = get_frontend_attributes(op3)
      self.assertEqual(attributes1.map.get("attr_a"), "a")
      self.assertEqual(attributes2.map.get("attr_a"), "c")
      self.assertEqual(attributes3.map.get("attr_a"), "a")
      self.assertIsNone(get_frontend_attributes(op4))

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
      attributes1 = get_frontend_attributes(op1)
      attributes2 = get_frontend_attributes(op2)
      attributes3 = get_frontend_attributes(op3)
      self.assertEqual(attributes1.map.get("attr_a"), "a")
      self.assertIsNone(attributes1.map.get("attr_b"))
      self.assertEqual(attributes2.map.get("attr_a"), "c")
      self.assertEqual(attributes2.map.get("attr_b"), "b")
      self.assertEqual(attributes3.map.get("attr_a"), "a")
      self.assertIsNone(attributes3.map.get("attr_b"))
      self.assertIsNone(get_frontend_attributes(op4))

  def testStochasticRounding(self):
    op1 = None
    op2 = None
    op3 = None
    op4 = None
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
      attributes1 = get_frontend_attributes(op1)
      attributes2 = get_frontend_attributes(op2)
      attributes3 = get_frontend_attributes(op3)
      attributes4 = get_frontend_attributes(op4)
      self.assertEqual(
          attributes1.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          "on")
      self.assertIsNone(attributes1.map.get("attr_b"))
      self.assertEqual(
          attributes2.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          "off")
      self.assertEqual(attributes2.map.get("attr_b"), "b")
      self.assertEqual(
          attributes3.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          "on")
      self.assertIsNone(attributes3.map.get("attr_b"))
      self.assertEqual(
          attributes4.map.get(
              backend_config_pb2.FrontendAttributeId.Name(
                  backend_config_pb2.FrontendAttributeId.STOCHASTIC_ROUNDING)),
          "off")
      self.assertIsNone(attributes4.map.get("attr_b"))


if __name__ == "__main__":
  googletest.main()

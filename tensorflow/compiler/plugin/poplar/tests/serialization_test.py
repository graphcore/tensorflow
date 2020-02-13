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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tempfile
import os
import re
import shutil
import numpy as np
import test_utils as tu

import tensorflow as tf
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import utils
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import types
from tensorflow.python.framework import tensor_spec


class FeedId:
  next_feed_id = 0

  @staticmethod
  def Next(name=None):
    result = "%s%d" % (name or 'feed', FeedId.next_feed_id)
    FeedId.next_feed_id += 1
    return result


def filesInFolder(folder):
  return [
      name for name in os.listdir(folder)
      if os.path.isfile(os.path.join(folder, name))
  ]


def PrimitiveTypeStringToNumpyDtype(primitive_type_str):
  primitive_type = xla_data_pb2.PrimitiveType.Value(primitive_type_str)
  return types.MAP_XLA_TYPE_TO_RECORD[primitive_type].numpy_dtype


class IpuSerializationTest(xla_test.XLATestCase):
  def _validateStreams(self,
                       streams,
                       expected_inputs,
                       expected_outputs=None,
                       expected_infeeds=None,
                       expected_outfeeds=None):
    inputs = streams.get("inputs", [])
    self.assertEqual(len(inputs), len(expected_inputs or []))
    for idx, stream in enumerate(inputs):
      expected_tensor, expected_type = expected_inputs[idx]
      self.assertEqual(idx, stream.get("index", -1))
      self.assertEqual(
          expected_tensor.dtype.as_numpy_dtype,
          PrimitiveTypeStringToNumpyDtype(stream.get("data_type")))
      self.assertEqual(expected_tensor.name, stream.get("name") + ":0")
      self.assertEqual(expected_type, stream.get("type"))
      self.assertEqual(expected_tensor.shape, stream.get("shape"))

    outputs = streams.get("outputs", [])
    self.assertEqual(len(outputs), len(expected_outputs or []))
    for idx, stream in enumerate(outputs):
      expected_tensor, expected_type = expected_outputs[idx]
      self.assertEqual(idx, stream.get("index", -1))
      self.assertEqual(
          expected_tensor.dtype.as_numpy_dtype,
          PrimitiveTypeStringToNumpyDtype(stream.get("data_type")))
      self.assertEqual(expected_tensor.name, stream.get("name") + ":0")
      self.assertEqual(expected_type, stream.get("type"))
      self.assertEqual(expected_tensor.shape, stream.get("shape"))

    infeeds = streams.get("infeeds", [])
    self.assertEqual(len(infeeds), len(expected_infeeds or []))
    for idx, stream in enumerate(infeeds):
      expected_tensor, expected_name = expected_infeeds[idx]
      self.assertEqual(idx, stream.get("index", -1))
      self.assertEqual(
          expected_tensor.dtype.as_numpy_dtype,
          PrimitiveTypeStringToNumpyDtype(stream.get("data_type")))
      self.assertEqual(expected_name, stream.get("name"))
      self.assertEqual(expected_tensor.shape, stream.get("shape"))

    outfeeds = streams.get("outfeeds", [])
    self.assertEqual(len(outfeeds), len(expected_outfeeds or []))
    for idx, stream in enumerate(outfeeds):
      expected_tensor, expected_name = expected_outfeeds[idx]
      self.assertEqual(idx, stream.get("index", -1))
      self.assertEqual(
          expected_tensor.dtype,
          PrimitiveTypeStringToNumpyDtype(stream.get("data_type")))
      self.assertEqual(expected_name, stream.get("name"))
      # First dimension is the number of tensors in the feed: ignore it
      self.assertEqual(list(expected_tensor.shape[1:]), stream.get("shape"))

  @test_util.deprecated_graph_mode_only
  def testSimpleFeedsInfoSerialization(self):
    if utils.running_on_ipu_model():
      self.skipTest(
          "Serialisation of executables is only supported for IPU targets")
    ndims = 2
    M = 3
    N = 5
    K = 7  # input features per group, output features per group, number of groups

    with self.session() as sess:

      def my_graph(inp, bias):
        with ops.device("/device:IPU:0"), variable_scope.variable_scope(
            "vs", use_resource=True, reuse=False):
          weights = variable_scope.get_variable("weights",
                                                [8] * ndims + [M, N * K])
          output = nn.convolution(inp,
                                  weights,
                                  strides=[1] + [4] * ndims + [1],
                                  padding="VALID",
                                  name='cnv')
          output = nn.bias_add(output, bias, name='bias_add')
          loss = math_ops.reduce_sum(math_ops.square(output))
          opt = gradient_descent.GradientDescentOptimizer(0.0005).minimize(
              loss)
          return loss, opt

      with ops.device("cpu"):
        inp = array_ops.placeholder(np.float32, [1] + [24] * ndims + [M * K],
                                    name="my/test/input_0")
        bias = array_ops.placeholder(np.float32, [N * K],
                                     name="my/test/bias/0")

      output = ipu.ipu_compiler.compile(my_graph, [inp, bias])

      with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "saved")
        if os.path.isdir(folder):
          shutil.rmtree(folder)

        tu.ReportJSON(self, sess, serialization_folder=folder)
        tu.move_variable_initialization_to_cpu()

        sess.run(variables.global_variables_initializer())
        sess.run(output, {inp: np.ones(inp.shape), bias: np.ones(bias.shape)})

        with variable_scope.variable_scope("vs", use_resource=True,
                                           reuse=True):
          weights = variable_scope.get_variable("weights")
        module_hash = None

        self.assertTrue(os.path.isdir(folder))
        for name in filesInFolder(folder):
          if not module_hash:
            m = re.match(r"([0-9a-f]{16})\..*", name)
            self.assertTrue(
                m, "Failed to identify module hash from filename %s" % name)
            module_hash = m.group(1)
          if name == module_hash + ".json":
            metadata = json.load(open(os.path.join(folder, name), "r"))
            self._validateStreams(
                metadata, [(bias, "input_data"), (inp, "input_data"),
                           (weights, "parameter")],
                [(tensor_spec.TensorSpec(shape=[],
                                         dtype=tf.float32,
                                         name="XLA_Retvals:0"), "input_data"),
                 (tensor_spec.TensorSpec(shape=[8, 8, 3, 35],
                                         dtype=tf.float32,
                                         name="XLA_Retvals:0"), "parameter")])
          else:
            self.assertTrue(
                name in [
                    "%s.ipu_bin" % module_hash,
                    "%s.ipu_bin.poplar_exec" % module_hash
                ], "Unexpected file generated: %s" % name)

  @test_util.deprecated_graph_mode_only
  def testInfeedsOutfeedInfoSerialization(self):
    if utils.running_on_ipu_model():
      self.skipTest(
          "Serialisation of executables is only supported for IPU targets")

    with self.session() as sess:
      dataset = tu.create_single_increasing_dataset(2, shape=[3, 3])
      infeed_name = FeedId.Next("feed")
      outfeed_name = FeedId.Next("feed")
      infeed_spec = dataset.element_spec[0]
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, infeed_name)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(outfeed_name)

      def body(const, inp):
        with variable_scope.variable_scope("vs", use_resource=True):
          inp2 = variable_scope.get_variable("input_2", [3, 3])
          v = inp * inp2 + const
          outfeed = outfeed_queue.enqueue(v)
          return (const, outfeed)

      def my_graph(const):
        return ipu.loops.repeat(4, body, (const), infeed_queue)

      with ops.device("cpu"):
        const = array_ops.placeholder(np.float32, [],
                                      name="my/test/constant/0")

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        output = ipu.ipu_compiler.compile(my_graph, inputs=[const])

      outfed = outfeed_queue.dequeue()
      with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "saved")
        if os.path.isdir(folder):
          shutil.rmtree(folder)

        tu.ReportJSON(self, sess, serialization_folder=folder)
        tu.move_variable_initialization_to_cpu()

        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())
        sess.run(output, {const: np.ones(const.shape)})
        outfed_result = sess.run(outfed)

        with variable_scope.variable_scope("vs", use_resource=True,
                                           reuse=True):
          inp2 = variable_scope.get_variable("input_2")
        module_hash = None

        self.assertTrue(os.path.isdir(folder))
        for name in filesInFolder(folder):
          if not module_hash:
            m = re.match(r"([0-9a-f]{16})\..*", name)
            self.assertTrue(
                m, "Failed to identify module hash from filename %s" % name)
            module_hash = m.group(1)
          if name == module_hash + ".json":
            with open(os.path.join(folder, name), "r") as metadata_file:
              metadata = json.load(metadata_file)
            self._validateStreams(
                metadata, [(const, "input_data"), (inp2, "parameter")],
                [(tensor_spec.TensorSpec(shape=[],
                                         dtype=tf.float32,
                                         name="XLA_Retvals:0"), "input_data")],
                [(infeed_spec, infeed_name)], [(outfed_result, outfeed_name)])
          else:
            self.assertTrue(
                name in [
                    "%s.ipu_bin" % module_hash,
                    "%s.ipu_bin.poplar_exec" % module_hash
                ], "Unexpected file generated: %s" % name)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

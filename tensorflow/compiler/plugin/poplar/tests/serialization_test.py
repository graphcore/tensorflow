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

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import types
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test
from tensorflow.python.ipu.config import IPUConfig


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
  def _configureIPU(self, serialization_folder):
    opts = IPUConfig()
    opts.device_connection.version = 'ipu1'
    opts.device_connection.type = utils.DeviceConnectionType.NEVER
    opts.serialization_output_folder = serialization_folder
    opts.configure_ipu_system()

  def _create_tmp_symlink(self, tmp_folder):
    tmp = "tempfiles"
    if os.path.islink(tmp):
      os.unlink(tmp)
    # Create a symlink in order for the serialization_folder to be identical for the different tests in the test suite.
    os.symlink(tmp_folder, tmp)
    return tmp

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
      self.assertEqual(
          expected_tensor.dtype.as_numpy_dtype,
          PrimitiveTypeStringToNumpyDtype(
              stream.get("shape", {}).get("data_type")))
      self.assertEqual(expected_tensor.name, stream.get("name") + ":0")
      self.assertEqual(expected_type, stream.get("type"))
      self.assertEqual(expected_tensor.shape,
                       stream.get("shape", {}).get("shape"))

    outputs = streams.get("outputs", [])
    self.assertEqual(len(outputs), len(expected_outputs or []))
    for idx, stream in enumerate(outputs):
      expected_tensor, expected_type = expected_outputs[idx]
      self.assertEqual(
          expected_tensor.dtype.as_numpy_dtype,
          PrimitiveTypeStringToNumpyDtype(
              stream.get("shape", {}).get("data_type")))
      self.assertEqual(expected_tensor.name, stream.get("name") + ":0")
      self.assertEqual(expected_type, stream.get("type"))
      self.assertEqual(expected_tensor.shape,
                       stream.get("shape", {}).get("shape"))

    infeeds = streams.get("infeeds", [])
    self.assertEqual(len(infeeds), len(expected_infeeds or []))
    for idx, infeed in enumerate(infeeds):
      expected_tensor, expected_name = expected_infeeds[idx]
      self.assertEqual(expected_name, infeed.get("name"))
      for stream_idx, stream in enumerate(infeed.get("streams")):
        self.assertEqual("%s.%d" % (expected_name, stream_idx),
                         stream.get("name"))
        self.assertEqual(
            expected_tensor.dtype.as_numpy_dtype,
            PrimitiveTypeStringToNumpyDtype(
                stream.get("shape", {}).get("data_type")))
        self.assertEqual(expected_tensor.shape,
                         stream.get("shape", {}).get("shape"))

    outfeeds = streams.get("outfeeds", [])
    self.assertEqual(len(outfeeds), len(expected_outfeeds or []))
    for idx, outfeed in enumerate(outfeeds):
      expected_tensor, expected_name = expected_outfeeds[idx]
      self.assertEqual(expected_name, outfeed.get("name"))
      for stream_idx, stream in enumerate(outfeed.get("streams")):
        self.assertEqual("%s.%d" % (expected_name, stream_idx),
                         stream.get("name"))
        self.assertEqual(
            expected_tensor.dtype,
            PrimitiveTypeStringToNumpyDtype(
                stream.get("shape", {}).get("data_type")))
        # First dimension is the number of tensors in the feed: ignore it
        self.assertEqual(list(expected_tensor.shape[1:]),
                         stream.get("shape", {}).get("shape"))

  def _find_feed_streams(self, feeds, feed_name):
    matches = [f for f in feeds if f["name"] == feed_name]
    self.assertEqual(len(matches), 1)
    self.assertTrue(len(matches[0].get("streams", [])) > 0)
    return matches[0]["streams"]

  def _validate_feeds(self, expected_feeds, test_feeds, auto_ids,
                      checkpoint_feeds, verification_info):
    self.assertEqual(len(expected_feeds), len(test_feeds))
    for feed, expected in expected_feeds.items():
      self.assertTrue(feed in checkpoint_feeds)
      streams = self._find_feed_streams(test_feeds, feed)
      seen_ids = []
      for stream in streams:
        info = verification_info.get(stream.get("handle"), {})
        self.assertEqual(expected.key, info.get("key"))
        self.assertTrue(isinstance(info.get("id"), int))
        if expected.start_id < 0:
          self.assertTrue(info["id"] not in auto_ids)
          auto_ids.append(info["id"])
        else:
          sid = info["id"]
          self.assertTrue(sid in range(expected.start_id, expected.start_id +
                                       len(streams)))
          self.assertTrue(sid not in seen_ids)
          seen_ids.append(sid)

  def _validate_tensors(self, expected, tensors, tensor_type, auto_ids,
                        verification_info):
    for tensor in [t for t in tensors if t.get("type") == tensor_type]:
      if "checkpoint" in tensor.get("name"):
        continue
      info = verification_info[tensor.get("handle")]
      self.assertEqual(expected.key, info.get("key"))
      self.assertTrue(isinstance(info.get("id"), int))
      if expected.start_id < 0:
        self.assertTrue(info["id"] not in auto_ids)
        auto_ids.append(info["id"])
      else:
        self.assertEqual(expected.start_id, info["id"])

  def _validate_secure_metadata(self, opts, metadata):
    auto_ids = []
    infeeds = metadata.get("infeeds", [])
    outfeeds = metadata.get("outfeeds", [])
    verification_info = metadata.get("verification_info", {})
    checkpoint_feeds = metadata.get("checkpoint", {}).get("feeds", [])
    self.assertEqual(
        len(opts.infeeds) + len(opts.outfeeds), len(checkpoint_feeds))

    self._validate_feeds(opts.infeeds, infeeds, auto_ids, checkpoint_feeds,
                         verification_info)
    self._validate_feeds(opts.outfeeds, outfeeds, auto_ids, checkpoint_feeds,
                         verification_info)
    self._validate_tensors(opts.inputs, metadata.get("inputs"), "input_data",
                           auto_ids, metadata.get("verification_info"))
    self._validate_tensors(opts.input_parameters, metadata.get("inputs"),
                           "parameter", auto_ids,
                           metadata.get("verification_info"))
    self._validate_tensors(opts.outputs, metadata.get("outputs"),
                           "parameter_out", auto_ids,
                           metadata.get("verification_info"))
    self._validate_tensors(opts.output_parameters, metadata.get("outputs"),
                           "output_data", auto_ids,
                           metadata.get("verification_info"))

  @test_util.deprecated_graph_mode_only
  def testSimpleFeedsInfoSerialization(self):
    ndims = 2
    M = 3
    N = 5
    K = 7  # input features per group, output features per group, number of groups

    # Disable the IPU model
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS",
                                  "").replace("--use_ipu_model", "")
    with test.mock.patch.dict(
        "os.environ",
        {"TF_POPLAR_FLAGS": poplar_flags}), self.session() as sess:

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

      with tempfile.TemporaryDirectory() as tmp_folder:
        tmp = self._create_tmp_symlink(tmp_folder)
        folder = os.path.join(tmp, "saved")

        self._configureIPU(folder)

        tu.move_variable_initialization_to_cpu()

        sess.run(variables.global_variables_initializer())
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "compilation only"):
          sess.run(output, {
              inp: np.ones(inp.shape),
              bias: np.ones(bias.shape)
          })

        with variable_scope.variable_scope("vs", use_resource=True,
                                           reuse=True):
          weights = variable_scope.get_variable("weights")
        module_hash = None

        self.assertTrue(os.path.isdir(folder))
        files = filesInFolder(folder)
        self.assertEqual(len(files), 2, "Expected 2 files, found: %s" % files)
        for name in files:
          if not module_hash:
            m = re.match(r"([0-9a-f]+)\..*", name)
            self.assertTrue(
                m, "Failed to identify module hash from filename %s" % name)
            module_hash = m.group(1)
          if name == module_hash + ".json":
            metadata = json.load(open(os.path.join(folder, name), "r"))
            self._validateStreams(
                metadata, [(bias, "input_data"), (inp, "input_data"),
                           (weights, "parameter")],
                [(tensor_spec.TensorSpec(shape=[],
                                         dtype=dtypes.float32,
                                         name="XLA_Retvals:0"), "output_data"),
                 (tensor_spec.TensorSpec(shape=[8, 8, 3, 35],
                                         dtype=dtypes.float32,
                                         name="vs/weights:0"), "parameter_out")
                 ])
          else:
            self.assertEqual(name, "%s.ipu_bin" % module_hash)

  @test_util.deprecated_graph_mode_only
  def testInfeedsOutfeedInfoSerialization(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS",
                                  "").replace("--use_ipu_model", "")
    with test.mock.patch.dict(
        "os.environ",
        {"TF_POPLAR_FLAGS": poplar_flags}), self.session() as sess:
      dataset = tu.create_single_increasing_dataset(2, shape=[3, 3])
      infeed_canonical_name = '1'
      outfeed_canonical_name = '2'
      infeed_spec = dataset.element_spec[0]
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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
      with tempfile.TemporaryDirectory() as tmp_folder:
        tmp = self._create_tmp_symlink(tmp_folder)
        folder = os.path.join(tmp, "saved")

        self._configureIPU(folder)

        tu.move_variable_initialization_to_cpu()

        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "compilation only"):
          sess.run(output, {const: np.ones(const.shape)})
        outfed_result = sess.run(outfed)

        with variable_scope.variable_scope("vs", use_resource=True,
                                           reuse=True):
          inp2 = variable_scope.get_variable("input_2")
        module_hash = None

        self.assertTrue(os.path.isdir(folder))
        files = filesInFolder(folder)
        self.assertEqual(len(files), 2, "Expected 2 files, found: %s" % files)
        for name in files:
          if not module_hash:
            m = re.match(r"([0-9a-f]+)\..*", name)
            self.assertTrue(
                m, "Failed to identify module hash from filename %s" % name)
            module_hash = m.group(1)
          if name == module_hash + ".json":
            with open(os.path.join(folder, name), "r") as metadata_file:
              metadata = json.load(metadata_file)
            self._validateStreams(metadata, [
                (const, "input_data"), (inp2, "parameter")
            ], [(tensor_spec.TensorSpec(shape=[],
                                        dtype=dtypes.float32,
                                        name="XLA_Retvals:0"), "output_data")],
                                  [(infeed_spec, infeed_canonical_name)],
                                  [(outfed_result, outfeed_canonical_name)])
          else:
            self.assertEqual(name, "%s.ipu_bin" % module_hash)

  @test_util.deprecated_graph_mode_only
  def testSimpleInfeedsDataSerialization(self):
    with self.session() as sess:
      num_elements = 10
      shape = (3, 5)
      dataset = tu.create_single_increasing_dataset(num_elements, shape=shape)
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      with tempfile.TemporaryDirectory() as tmp_folder:
        output_folder = self._create_tmp_symlink(tmp_folder)
        output_file = os.path.join(output_folder, "infeed.json")

        sess.run(infeed_queue.initializer)

        utils.export_dataset_to_file(infeed_queue, output_file, num_elements)

        files = filesInFolder(output_folder)
        self.assertEqual(len(files), 1, "Expected 1 file, found: %s" % files)

  @test_util.deprecated_graph_mode_only
  def testSimpleDatasetDataSerialization(self):
    num_elements = 10
    shape = (3, 5)
    dataset = tu.create_single_increasing_dataset(num_elements, shape=shape)

    with tempfile.TemporaryDirectory() as tmp_folder:
      output_folder = self._create_tmp_symlink(tmp_folder)
      output_file = os.path.join(output_folder, "dataset.json")

      utils.export_dataset_to_file(dataset, output_file, num_elements)

      files = filesInFolder(output_folder)
      self.assertEqual(len(files), 1, "Expected 1 file, found: %s" % files)

  @test_util.deprecated_graph_mode_only
  def testTupleInfeedsDataSerialization(self):
    with self.session() as sess:
      num_elements = 10
      shape = (4, 8)
      shape_2 = (2, 4, 2, 2)
      dataset = tu.create_single_increasing_dataset(num_elements, shape=shape)

      def dataset_parser(value):
        image_1 = value
        image_2 = (value + 10.) / 2.0
        return (image_1, array_ops.reshape(image_2, shape_2))

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      with tempfile.TemporaryDirectory() as tmp_folder:
        output_folder = self._create_tmp_symlink(tmp_folder)
        output_file = os.path.join(output_folder, "infeed.bin")

        sess.run(infeed_queue.initializer)

        utils.export_dataset_to_file(infeed_queue, output_file, num_elements)

        files = filesInFolder(output_folder)
        self.assertEqual(
            len(files), 1,
            "Expected 1 file (Containing both tuple elements), found: %s" %
            files)

  @test_util.deprecated_graph_mode_only
  def testNamedInfeedsDataSerialization(self):
    with self.session() as sess:
      num_elements = 10
      shape = (4, 8)
      shape_2 = (2, 4, 2, 2)
      dataset = tu.create_single_increasing_dataset(num_elements, shape=shape)

      def dataset_parser(value):
        image_1 = value
        image_2 = (value + 10.) / 2.0
        return {"a": image_1, "b": array_ops.reshape(image_2, shape_2)}

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      with tempfile.TemporaryDirectory() as tmp_folder:
        output_folder = self._create_tmp_symlink(tmp_folder)
        output_file = os.path.join(output_folder, "infeed.bin")

        sess.run(infeed_queue.initializer)

        utils.export_dataset_to_file(infeed_queue, output_file, num_elements)

        files = filesInFolder(output_folder)
        self.assertEqual(
            len(files), 1,
            "Expected 1 file containing both feed 'a', and feed 'b', found: %s"
            % files)

  @test_util.deprecated_graph_mode_only
  def testNamedInfeedsDataSerializationStep(self):
    with self.session() as sess:
      num_elements = 1000
      shape = (224, 224, 3)
      shape_2 = (224, 3, 224)
      dataset = tu.create_single_increasing_dataset(num_elements, shape=shape)

      def dataset_parser(value):
        image_1 = value
        image_2 = (value + 10.) / 2.0
        return {"a": image_1, "b": array_ops.reshape(image_2, shape_2)}

      dataset = dataset.map(dataset_parser)
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      with tempfile.TemporaryDirectory() as tmp_folder:
        output_folder = self._create_tmp_symlink(tmp_folder)
        output_file = os.path.join(output_folder, "infeed.bin")

        sess.run(infeed_queue.initializer)

        utils.export_dataset_to_file(infeed_queue, output_file, num_elements)

        files = filesInFolder(output_folder)
        self.assertEqual(
            len(files), 1,
            "Expected 1 file containing both feed 'a', and feed 'b', found: %s"
            % files)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

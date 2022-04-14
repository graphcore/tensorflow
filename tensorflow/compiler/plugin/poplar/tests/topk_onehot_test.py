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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


class OneHotTopK(xla_test.XLATestCase):
  def testOneHot(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def executeModel(inputs, expected):
      with self.session() as sess:

        # Decide what the output type should be.
        data_type = inputs["on"].dtype

        # The actual model function which perfoms the one-hot operation based on the inputs given to executeModel.
        def model(a):
          return array_ops.one_hot(a,
                                   inputs["n_classes"],
                                   dtype=data_type,
                                   on_value=inputs["on"],
                                   off_value=inputs["off"],
                                   axis=inputs["axis"])

        # We run once on the CPU to get the expected result, then on the IPU to compare the two.
        cpuRun = expected is None

        with ops.device('cpu'):
          pa = array_ops.placeholder(np.int32, inputs["shape"], name="a")

        # Check if we should be running on IPU or cpu.
        device = "cpu:0" if cpuRun else "/device:IPU:0"

        with ops.device(device):
          out = model(pa)

        in_data = np.array(inputs["in_values"])

        fd = {pa: in_data}
        result = sess.run(out, fd)

        if cpuRun:
          return result
        self.assertAllClose(result, expected)

    # Generate a multi dimensional matrix.
    largish_matrix_size = [4, 3, 4, 2, 2]
    largish_matrix_data = np.random.randint(1, np.prod(largish_matrix_size),
                                            largish_matrix_size)

    # Generate a vector as well, as using just the matrix will increase test times unnecessarily
    vector_size = [4, 3, 4, 2, 2]
    vector_data = np.random.randint(1, np.prod(largish_matrix_size),
                                    largish_matrix_size)

    inputs = [
        # Test different dimensions.
        {
            "n_classes": 10,
            "shape": [4],
            "in_values": [1, 2, 3, 4],
            "on": np.float32(2.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 1200,
            "shape": [4, 2],
            "in_values": [[1, 1], [2, 5], [4, 3], [4, 6]],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 1200,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        # Test different depths
        {
            "n_classes": 1,
            "shape": [4],
            "in_values": [1, 2, 3, 4],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 12000,
            "shape": [4, 2],
            "in_values": [[1, 1], [2, 5], [4, 3], [4, 6]],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },

        # Test different axes.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 0
        },
        {
            "n_classes": 1200,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 3
        },
        {
            "n_classes": 100,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 2
        },
        # Test different on/off.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(0.25),
            "off": np.float32(0.1),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(20.0),
            "off": np.float32(-1.0),
            "axis": 0
        },
        # Float16 is the only data type we will run on assembly so we have specific cases for that.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float16(1.0),
            "off": np.float16(0.0),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float16(2.0),
            "off": np.float16(3.0),
            "axis": 1
        },

        # Check int32 works as well
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.int32(4.0),
            "off": np.int32(2.0),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.int32(4.0),
            "off": np.int32(2.0),
            "axis": 1
        },
    ]

    for test_case in inputs:
      # Run on CPU first
      result = executeModel(test_case, None)

      # Run on IPU and test against CPU out.
      executeModel(test_case, result)

  def testTopK(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    def doTestTopK(self, dtype):
      with self.session() as sess:

        n_categories = 1200
        topn = 24

        def model(a):
          _, indices = nn.top_k(a, topn)
          return indices

        with ops.device('cpu'):
          pa = array_ops.placeholder(dtype, [n_categories], name="a")

        with ops.device("/device:IPU:0"):
          out = model(pa)

        report_json = tu.ReportJSON(self, sess)
        report_json.reset()

        # Shuffled set of values of specified dtype in [0:n_categories).
        # This ensures there is a single unique sort result.
        pa_input = np.arange(n_categories, dtype=dtype)
        np.random.shuffle(pa_input)
        expected = (-pa_input).argsort()[:topn]

        fd = {pa: pa_input}
        result = sess.run(out, fd)
        self.assertAllClose(result, expected)

        report_json.parse_log(assert_len=4)

    testTypes = [np.float16, np.float32, np.int32]
    for dtype in testTypes:
      doTestTopK(self, dtype)

  def testInTopK(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:

      batchsize = 4
      n_categories = 1200
      topn = 8

      def model(a, b):
        return nn.in_top_k(a, b, topn)

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.float32, [batchsize, n_categories])
        pb = array_ops.placeholder(np.int32, [batchsize])

      with ops.device("/device:IPU:0"):
        out = model(pa, pb)

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()

      input = np.random.rand(batchsize, n_categories)
      input = input / np.sqrt(np.sum(input**2))

      ref = (-input).argsort(axis=1)[:, :1]
      ref = ref.reshape([batchsize])

      fd = {pa: input, pb: ref}
      result = sess.run(out, fd)
      self.assertAllClose(result, [True, True, True, True])

      report_json.parse_log(assert_len=4)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

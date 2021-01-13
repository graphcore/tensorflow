# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
import os
import re
import shutil
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

# You should not run multiple instances of this test in parallel.
# For example bazel test ... --runs_per_test 10.
# It will fail as there is race for framework.json file.


def check_if_directory_exists(dir_subname, cwd_files):
  is_dir = False
  dir_name = ""
  for file_name in cwd_files:
    if bool(re.search(dir_subname, file_name)):
      is_dir = True
      dir_name = file_name
  return is_dir, dir_name


def is_json(test_str):
  test_s = json.dumps(test_str)
  try:
    json_object = json.loads(test_s)
  except ValueError:
    return False, json_object
  return True, json_object


# pylint: disable=abstract-method
class AutoReportDirTest(xla_test.XLATestCase):
  def testIfReportDirCreated0(self):
    poplar_engine_opts_flags = os.environ.get("POPLAR_ENGINE_OPTIONS", "")
    poplar_engine_opts_flags += "{\"autoReport.all\":\"true\"}"

    with test.mock.patch.dict(
        "os.environ", {"POPLAR_ENGINE_OPTIONS": poplar_engine_opts_flags}):

      with self.session() as sess:

        def simple_net(x):
          return math_ops.square(x, name="name_simple_net")

        with ops.device('cpu'):
          x1 = array_ops.placeholder(dtypes.int32, [2, 2])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r1 = ipu.ipu_compiler.compile(simple_net, inputs=[x1])
          i_x1 = np.full((2, 2), 10)

          cfg = ipu.utils.create_ipu_config(profiling=True)
          ipu.utils.configure_ipu_system(cfg)

          sess.run(r1, {x1: i_x1})

      cwd = os.getcwd()
      cwd_files = os.listdir(cwd)
      dir_subname = 'cluster_'
      is_dir, dir_name = check_if_directory_exists(dir_subname, cwd_files)
      self.assertTrue(is_dir)
      if is_dir:
        cluster_path = os.path.abspath(dir_name)
        cluster_dir_files = os.listdir(cluster_path)
        self.assertTrue("framework.json" in cluster_dir_files)
        test_file = cluster_path + "/framework.json"
        with open(test_file) as f:
          json_txt = json.load(f)
          is_j, _ = is_json(json_txt)
          self.assertTrue(is_j)
        shutil.rmtree(cluster_path)

  def testIfReportDirCreated1(self):
    poplar_engine_opts_flags = os.environ.get("POPLAR_ENGINE_OPTIONS", "")
    # pylint: disable=line-too-long
    poplar_engine_opts_flags += "{\"autoReport.all\":\"true\", \"autoReport.directory\":\"./tommyFlowers\"}"
    # pylint: enable=line-too-long

    with test.mock.patch.dict(
        "os.environ", {"POPLAR_ENGINE_OPTIONS": poplar_engine_opts_flags}):

      with self.session() as sess:

        def simple_net(x):
          return math_ops.square(x, name="name_simple_net")

        with ops.device('cpu'):
          x1 = array_ops.placeholder(dtypes.int32, [2, 2])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r1 = ipu.ipu_compiler.compile(simple_net, inputs=[x1])
          i_x1 = np.full((2, 2), 10)

          cfg = ipu.utils.create_ipu_config(profiling=True)
          ipu.utils.configure_ipu_system(cfg)

          sess.run(r1, {x1: i_x1})

      cwd = os.getcwd()
      cwd_files = os.listdir(cwd)
      dir_subname = 'tommyFlowers'
      is_dir, dir_name = check_if_directory_exists(dir_subname, cwd_files)
      self.assertTrue(is_dir)
      if is_dir:
        cluster_path = os.path.abspath(dir_name)
        cluster_dir_files = os.listdir(cluster_path)
        self.assertTrue("framework.json" in cluster_dir_files)
        test_file = cluster_path + "/framework.json"
        with open(test_file) as f:
          json_txt = json.load(f)
          is_j, _ = is_json(json_txt)
          self.assertTrue(is_j)
        shutil.rmtree(cluster_path)

  def testIfReportDirNotCreated(self):
    poplar_engine_opts_flags = os.environ.get("POPLAR_ENGINE_OPTIONS", "")
    poplar_engine_opts_flags += "{}"

    with test.mock.patch.dict(
        "os.environ", {"POPLAR_ENGINE_OPTIONS": poplar_engine_opts_flags}):

      with self.session() as sess:

        def simple_net(x):
          return math_ops.square(x, name="name_simple_net")

        with ops.device('cpu'):
          x1 = array_ops.placeholder(dtypes.int32, [2, 2])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r1 = ipu.ipu_compiler.compile(simple_net, inputs=[x1])
          i_x1 = np.full((2, 2), 10)

          cfg = ipu.utils.create_ipu_config(profiling=True)
          ipu.utils.configure_ipu_system(cfg)

          sess.run(r1, {x1: i_x1})

      cwd = os.getcwd()
      cwd_files = os.listdir(cwd)
      dir_subname = 'cluster_'
      is_dir, _ = check_if_directory_exists(dir_subname, cwd_files)
      self.assertFalse(is_dir)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

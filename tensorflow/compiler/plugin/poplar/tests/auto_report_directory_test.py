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
from pathlib import Path
import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

REPORT_DIR_PREFIX = "tf_report_"


def find_files_by_substring(directory, substring):
  return [f for f in Path(directory).iterdir() if substring in f.name]


def is_json(test_str):
  test_s = json.dumps(test_str)
  try:
    json.loads(test_s)
  except json.decoder.JSONDecodeError:
    return False
  return True


# Helper to run a simple graph on the IPU. It doesn't really matter what it is
def createSimpleGraph():
  def simple_net(x):
    return math_ops.square(x)

  with ops.device("cpu"):
    x = array_ops.placeholder(dtypes.int32, [2, 2])

  with ipu.scopes.ipu_scope("/device:IPU:0"):
    run_op = ipu.ipu_compiler.compile(simple_net, inputs=[x])

  return run_op, x


def add_to_poplar_engine_options(new_opts):
  cur_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
  return {"POPLAR_ENGINE_OPTIONS": json.dumps({**cur_opts, **new_opts})}


# pylint: disable=abstract-method
class AutoReportDirTest(xla_test.XLATestCase):
  def setUp(self):
    super().setUp()
    # Temporarily move to a temp dir
    os.chdir(self.get_temp_dir())

  def testAutoReportDirNotCreated(self):
    with test.mock.patch.dict("os.environ", add_to_poplar_engine_options({})):

      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with self.session() as sess:
        run_graph_op, x = createSimpleGraph()
        sess.run(run_graph_op, {x: np.full((2, 2), 10)})

      # Make sure there's no report directories in the cwd
      repdirs = find_files_by_substring(os.getcwd(), REPORT_DIR_PREFIX)
      self.assertTrue(repdirs == [])

  def testAutoReportDirAutoCreated(self):
    with test.mock.patch.dict(
        "os.environ", add_to_poplar_engine_options({"autoReport.all":
                                                    "true"})):
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with self.session() as sess:
        run_graph_op, x = createSimpleGraph()
        sess.run(run_graph_op, {x: np.full((2, 2), 10)})

      # Make sure there is at least one report directory in the cwd
      repdirs = find_files_by_substring(os.getcwd(), REPORT_DIR_PREFIX)
      self.assertTrue(repdirs)

      # Make sure there's a JSON framework.json in all of them
      for repdir in repdirs:
        framework_file = find_files_by_substring(repdir, "framework.json")
        self.assertTrue(framework_file)
        with framework_file[0].open() as fp:
          self.assertTrue(is_json(json.load(fp)))

  def testAutoReportDirCreatedWhenSpecified(self):
    with test.mock.patch.dict(
        "os.environ",
        add_to_poplar_engine_options({
            "autoReport.all": "true",
            "autoReport.directory": "./tommyFlowers"
        })):

      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with self.session() as sess:
        run_graph_op, x = createSimpleGraph()
        sess.run(run_graph_op, {x: np.full((2, 2), 10)})

      # Make sure there's exactly one report directory with the right name
      rootdirs = find_files_by_substring(os.getcwd(), "tommyFlowers")
      self.assertTrue(rootdirs)
      self.assertTrue(len(rootdirs) == 1)
      rootdir = rootdirs[0]

      # for each report directory - must be at least 1
      repdirs = find_files_by_substring(rootdir, REPORT_DIR_PREFIX)
      self.assertTrue(repdirs)
      self.assertTrue(len(repdirs) >= 1)
      repdir = repdirs[0]

      # Make sure there's a JSON framework.json in it
      framework_file = find_files_by_substring(repdir, "framework.json")
      self.assertTrue(framework_file)
      with framework_file[0].open() as fp:
        self.assertTrue(is_json(json.load(fp)))

  def testAutoGeneratedDirectoryReused(self):
    with test.mock.patch.dict(
        "os.environ", add_to_poplar_engine_options({"autoReport.all":
                                                    "true"})):
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with self.session() as sess:
        run_graph_op, x = createSimpleGraph()
        sess.run(run_graph_op, {x: np.full((2, 2), 10)})

        # Make sure there is at least one report directory in the cwd
        repdirs = find_files_by_substring(os.getcwd(), REPORT_DIR_PREFIX)
        self.assertTrue(repdirs)
        num_dirs = len(repdirs)

        # Run it again, the report directories should be re-used
        sess.run(run_graph_op, {x: np.full((2, 2), 10)})
        repdirs = find_files_by_substring(os.getcwd(), REPORT_DIR_PREFIX)
        self.assertTrue(len(repdirs) == num_dirs)

  def testAutoAssignReportSubdirectoriesAllowsMultipleReports(self):
    report_helper = tu.ReportHelper()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      run_graph_op_1, x_1 = createSimpleGraph()
      sess.run(run_graph_op_1, {x_1: np.full((2, 2), 5)})
      # Assert one report generated.
      self.assert_num_reports(report_helper, 1)

      sess.run(gen_ipu_ops.ipu_clear_all_xla_compilation_caches())

      run_graph_op_2, x_2 = createSimpleGraph()
      sess.run(run_graph_op_2, {x_2: np.full((2, 2), 10)})
      # Assert second report does not override first.
      self.assert_num_reports(report_helper, 2)

  def testAutoAssignReportSubdirectoriesSubdirectoryReused(self):
    report_helper = tu.ReportHelper()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      run_graph_op_1, x_1 = createSimpleGraph()
      sess.run(run_graph_op_1, {x_1: np.full((2, 2), 5)})
      # Assert one report generated.
      self.assert_num_reports(report_helper, 1)

      sess.run(run_graph_op_1, {x_1: np.full((2, 2), 5)})
      # Assert report from rerun overrides report from previous run.
      self.assert_num_reports(report_helper, 1)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()

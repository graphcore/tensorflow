# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import collections
import contextlib
import fnmatch
from functools import reduce
import json as js
import re
import os
import pathlib
import shutil
import tempfile
import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig


def compute_device_count(pipelining=False, sharded=False, replicated=False):
  if sharded:
    device_count = 2 * (2 if replicated else 1)
  elif pipelining:
    device_count = 4 * (2 if replicated else 1)
  else:
    device_count = 2 if replicated else 0

  return device_count


@contextlib.contextmanager
def ipu_session(disable_grappler_optimizers=None):
  config = None
  # Disable any requested grappler optimizers
  if disable_grappler_optimizers and \
      isinstance(disable_grappler_optimizers, list):
    config = config_pb2.ConfigProto()
    for opt in disable_grappler_optimizers:
      assert hasattr(config.graph_options.rewrite_options, opt), \
          f"Tried to disable grappler optimizer '{opt}' but it's not an" \
          " attribute of the RewriterConfig proto"
      setattr(config.graph_options.rewrite_options, opt,
              rewriter_config_pb2.RewriterConfig.OFF)
  with session_lib.Session(config=config) as sess:
    yield sess


def items_matching_at_least_one_pattern(items, patterns):
  matches = []
  patterns = [x + '*' for x in patterns]
  for item in items:
    if [p for p in patterns if fnmatch.fnmatch(item, p)]:
      matches.append(item)
  return matches


def names_in_blacklist(names, blacklist):
  return items_matching_at_least_one_pattern(names, blacklist)


def missing_names_in_whitelist_entries(names, whitelist):
  fail_list = []
  wl = ['*' + x + '*' for x in whitelist]
  for name in names:
    if name and not [x for x in wl if fnmatch.fnmatch(name, x)]:
      fail_list += [name]
  return fail_list


def missing_whitelist_entries_in_names(names, whitelist):
  fail_list = []
  wl = ['*' + x + '*' for x in whitelist]
  for x in wl:
    if not [name for name in names if fnmatch.fnmatch(name, x)]:
      fail_list += [x]
  return fail_list


def count_matches_in_list(input_list, to_match):
  return len([s for s in input_list if fnmatch.fnmatch(s, to_match)])


class TensorMap(object):
  class Tile(object):
    def __init__(self, tile, num_elements):
      self.tile = tile
      self.num_elements = num_elements

    def __eq__(self, other):
      return self.tile == other.tile and self.num_elements == other.num_elements

  class Tensor(object):
    def __init__(self, inst, index, shape, dtype, has_constant, has_aliases,
                 num_elements, tiles, name):
      self.inst = inst
      self.index = index
      self.shape = shape
      self.dtype = dtype
      self.has_constant = has_constant
      self.has_aliases = has_aliases
      self.num_elements = num_elements
      self.tiles = tiles
      self.name = name

    def tile_ids(self):
      return list({t.tile for t in self.tiles})

    @property
    def id(self):
      return "%s,%d" % (self.inst, self.index)

  def __init__(self, tensor_map, num_tiles_per_ipu):
    self.num_tiles_per_ipu = num_tiles_per_ipu
    self.mappings = {}
    for comp, js_tensors in tensor_map["mappings"].items():
      tensors = []
      for js_tensor in js_tensors:
        tiles = []
        for tile in js_tensor[7]:
          assert len(tile) == 2
          tiles.append(TensorMap.Tile(tile[0], tile[1]))
        tensors.append(
            TensorMap.Tensor(inst=js_tensor[0],
                             index=js_tensor[1],
                             shape=js_tensor[2],
                             dtype=js_tensor[3],
                             has_constant=bool(js_tensor[4]),
                             has_aliases=bool(js_tensor[5]),
                             num_elements=js_tensor[6],
                             tiles=tiles,
                             name=js_tensor[8]))
      self.mappings[comp] = tensors

  def all_tensors(self):
    for _, tensors in self.mappings.items():
      for tensor in tensors:
        yield tensor

  def tile_ids(self, computation=None):
    if isinstance(computation, list):
      computations = computation
    else:
      computations = [computation] if computation else self.mappings.keys()
    ids = set()
    for c in computations:
      for tensor in self.mappings[c]:
        ids.update(tensor.tile_ids())
    return ids

  def ipu_ids(self, computation=None):
    tile_ids = self.tile_ids(computation)
    return {int(tile_id / self.num_tiles_per_ipu) for tile_id in tile_ids}

  def computation_names(self):
    return list(self.mappings.keys())

  def tensor_inst_name_mappings(self):
    mappings = {}
    for comp, tensors in self.mappings.items():
      for tensor in tensors:
        assert tensor.id not in mappings, ("Instruction %s already in"
                                           " mappings %s") % (tensor.id,
                                                              mappings)
        mappings[tensor.id] = tensor.name
    return mappings


# Members of this class are attached to TensorFlowTestCase.
class TestCaseExtensions(object):
  def _assert_all_in_tolerance(self, actual, expected, tolerance=0.001):
    """Asserts that all values are within relative tolerance of expected.
    Only intended to be used with integer values.
    """
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.assertAllInRange(actual, low, high)

  def assert_all_compute_sets_and_list(self, report, ok):
    """Asserts all the compute sets match a pattern in the whitelist and also
    asserts that all the whitelist patterns match at least one compute set.
    """
    not_in_whitelist = []
    not_in_report = []
    whitelist = ['*' + x + '*' for x in ok]

    for expected in whitelist:
      if not any(
          fnmatch.fnmatch(actual.name, expected)
          for actual in report.compilation.computeSets):
        not_in_report.append(expected)
    for actual in report.compilation.computeSets:
      if not any(
          fnmatch.fnmatch(actual.name, expected) for expected in whitelist):
        not_in_whitelist.append(actual.name)

    error_msg = "\n"
    if not_in_report:
      error_msg = "Whitelist items [%s] not found in compute sets:\n\t%s" % (
          ",".join(not_in_report), "\n\t".join(
              cs.name for cs in report.compilation.computeSets))
    if not_in_report and not_in_whitelist:
      error_msg += "\n"
    if not_in_whitelist:
      error_msg += "Compute sets items [%s] not found in whitelist:\n\t%s" % (
          ",".join(not_in_whitelist), "\n\t".join(ok))

    self.assertFalse(not_in_report + not_in_whitelist, error_msg)

  def assert_compute_sets_matches(self, report, expr, num_matches, msg=None):
    """Asserts the number of compute sets in the report which match expr."""
    cs_names = (cs.name for cs in report.compilation.computeSets)
    self.assertEqual(count_matches_in_list(cs_names, expr), num_matches, msg)

  def assert_compute_sets_contain_list(self, report, ok):
    """Asserts that of all the whitelist patterns match at least one compute
    set in the report.
    """
    whitelist = ['*' + x + '*' for x in ok]
    not_in_report = []
    for expected in whitelist:
      if not any(
          fnmatch.fnmatch(actual.name, expected)
          for actual in report.compilation.computeSets):
        not_in_report.append(expected)

    self.assertFalse(
        not_in_report,
        "Whitelist items [%s] not found in compute sets:\n\t%s" %
        (",".join(not_in_report), "\n\t".join(
            cs.name for cs in report.compilation.computeSets)))

  def assert_compute_sets_not_in_blacklist(self, report, blacklist):
    """Asserts that no compute sets in the report match any of the blacklist
    items.
    """
    blacklist = ['*' + x + '*' for x in blacklist]
    in_report = []
    for item in blacklist:
      if any(
          fnmatch.fnmatch(cs.name, item)
          for cs in report.compilation.computeSets):
        in_report.append(item)

    self.assertFalse(
        in_report, "Blacklist items [%s] found in compute sets:\n\t%s" %
        (",".join(in_report), "\n\t".join(
            cs.name for cs in report.compilation.computeSets)))

  def assert_each_tile_memory_is_less_than(self,
                                           report,
                                           max_expected,
                                           tolerance=0.01):
    """Assert total memory (excluding gaps) on each tile is below max."""
    high = int(max_expected * (1.0 + tolerance))
    tile_memory = [
        tile.memory.total.excludingGaps for tile in report.compilation.tiles
    ]
    self.assertAllInRange(tile_memory, 0, high)

  def assert_total_tile_memory(self, report, expected, tolerance=0.01):
    """Assert total memory (excluding gaps) across all tiles is close to
    expected.
    """
    total_memory = sum(tile.memory.total.excludingGaps
                       for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([total_memory], expected, tolerance)

  def assert_max_tile_memory(self, report, expected, tolerance=0.01):
    """Assert peak tile memory (excluding gaps) is close to expected."""
    max_memory = max(tile.memory.total.excludingGaps
                     for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([max_memory], expected, tolerance)

  def assert_always_live_memory(self, report, expected, tolerance=0.01):
    """Assert total always-live memory across all tiles is close to
    expected.
    """
    always_live_memory = sum(tile.memory.alwaysLiveBytes
                             for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([always_live_memory], expected, tolerance)

  def assert_execution_report_cycles(self,
                                     report,
                                     idx,
                                     expected,
                                     tolerance=0.01):
    """Asserts the total cycles on each tile are close to expected for the
    specified execution.
    """
    steps = report.execution.runs[idx].steps
    n_tiles = len(steps[0].cyclesByTile)
    cycles = [
        sum(step.cyclesByTile[t] for step in steps) for t in range(n_tiles)
    ]
    self._assert_all_in_tolerance(cycles, expected, tolerance)

  def assert_number_of_executions(self, report, n):
    """Asserts the number of executions in the report."""
    self.assertLen(report.execution.runs, n)


# Attach everything from TestCaseExtensions onto TensorFlowTestCase.
# Note that XLATestCase inherits from TensorFlowTestCase.
for attr in dir(TestCaseExtensions):
  if not attr.startswith('__'):
    setattr(TensorFlowTestCase, attr, getattr(TestCaseExtensions, attr))


class ReportHelper():
  """ ReportHelper creates a temporary directory for reports to be generated
  in. `set_autoreport_options` configures poplar to use this directory.

  Reports are generated in unique subdirectories of the temporary directory and
  can be found by calling `find_report` or `find_reports`.

  Reports can also be cleared by calling `clear_reports`.

  All files are automatically cleaned up when the object is destroyed.
  """
  def __init__(self):
    self._directory = tempfile.mkdtemp(prefix=f"tf_")
    # Used to give a better error message if no reports were generated because
    # you forgot to call set_autoreport_options.
    self._set_options_called = False

  def _find_report_subdirectories(self):
    # Find all subdirectories in the report directory.
    directory = pathlib.Path(self._directory)
    if not directory.exists():
      if not self._set_options_called:
        raise RuntimeError("To use this helper you must setup the poplar " +
                           "autoReport options with set_autoreport_options.")
      raise IOError(
          f"Report directory does not exist: {self._directory}\nEither " +
          "no reports have been generated or the directory was deleted.")
    return directory.glob('tf_report_*/')

  def _find_report_files_in_subdirectory(self, directory):
    return directory.glob("*.pop")

  def set_autoreport_options(self,
                             cfg,
                             *,
                             output_graph_profile=True,
                             output_execution_profile=False,
                             max_execution_reports=1000):
    """Sets autoReport engine options in the IPUConfig.

    Set outputExecutionProfile to True to allow execution reports to be
    generated.

    If execution reports are enabled, max_execution_reports controls the
    maximum number of executions included in a report.
    """
    self._set_options_called = True
    options = {
        "autoReport.directory": self._directory,
        "autoReport.outputGraphProfile": str(output_graph_profile).lower(),
        "autoReport.outputExecutionProfile":
        str(output_execution_profile).lower(),
        "autoReport.executionProfileProgramRunCount":
        str(max_execution_reports),
    }
    cfg.compilation_poplar_options = options
    cfg._profiling.auto_assign_report_subdirectories = True  # pylint: disable=protected-access

  def assert_num_reports(self, n):
    """Asserts the number of reports found matches the number given."""
    num_reports = len(self.find_reports())
    assert num_reports == n, f"Expected {n} report(s) but found {num_reports}"

  def find_reports(self):
    """Finds and returns the paths to generated report files in order of
    creation time (oldest first).
    """
    paths = []
    for d in self._find_report_subdirectories():
      files_ = list(self._find_report_files_in_subdirectory(d))
      # Only expect 1 report file per report subdirectory.
      if len(files_) != 1:
        raise IOError(f"Expected 1 report file in each report " +
                      f"subdirectory but found {len(files_)} in {d}:" +
                      "".join(f"\n   {f.name}" for f in files_))
      # Add report file absolute path to result.
      paths.append(str(files_[0]))

    # Sort by oldest first
    paths.sort(key=lambda p: os.stat(p).st_ctime)
    return paths

  def find_report(self):
    """Finds and returns the paths to the generated report file.
    Asserts the only one report has been generated.
    """
    reports = self.find_reports()
    num_reports = len(reports)
    assert num_reports == 1, f"Expected 1 report but found {num_reports}"
    return reports[0]

  def clear_reports(self):
    """Clears all existing reports and their subdirectories."""
    # Remove the whole directory and recreate it rather than removing each
    # subdirectory individually.
    shutil.rmtree(self._directory)
    os.mkdir(self._directory)

  # Automatically clean up all files when this instance is destroyed.
  def __del__(self):
    # Ignore errors to clean up as much as possible.
    shutil.rmtree(self._directory, ignore_errors=True)


class ReportJSON(object):
  def __init__(self,
               test,
               sess=None,
               profiling=True,
               compile_ipu_code=False,
               tiles_per_ipu=0,
               device_count_override=None,
               execution_trace=True,
               sharded=False,
               pipelining=False,
               configure_device=True,
               replicated=False,
               max_cross_replica_sum_buffer_size=0,
               max_inter_ipu_copies_buffer_size=0,
               merge_infeed_io_copies=False,
               always_rearrange_copies_on_the_host=False,
               serialization_folder="",
               estimator_hook=False,
               eager_mode=False,
               allow_recompute=False,
               use_stable_norm_statistics=False,
               set_opts_fn=None,
               triangular_solve_expander_block_size=0,
               minimum_remote_tensor_size=128,
               use_hw=False,
               num_io_tiles=0):
    self.report = None
    self.test = test
    self.sess = sess
    self.eager_mode = eager_mode

    assert not eager_mode or not sess, "Sessions can't be used in eager mode"

    # If no session is passed to the constructor then assume
    # the events will be provided by the user.
    if sess:
      self.create_ipu_event_trace()
    if (sess or estimator_hook or eager_mode) and configure_device:
      # yapf: disable
      assert not (pipelining and device_count_override
                 ), "Can't have both pipelining enabled and device_count_override"
      assert not (sharded and device_count_override
                 ), "Can't have both sharded enabled and device_count_override"
      # yapf: enable

      opts = IPUConfig()
      opts._profiling.profiling = profiling  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = False  # pylint: disable=protected-access
      opts._profiling.use_poplar_cbor_report = False  # pylint: disable=protected-access
      opts._profiling.profile_execution = execution_trace  # pylint: disable=protected-access
      opts.experimental.always_rearrange_copies_on_the_host = \
          always_rearrange_copies_on_the_host
      opts.optimizations.merge_infeed_io_copies = merge_infeed_io_copies

      opts.optimizations.maximum_cross_replica_sum_buffer_size = \
          max_cross_replica_sum_buffer_size
      opts.optimizations.maximum_inter_ipu_copies_buffer_size = \
          max_inter_ipu_copies_buffer_size
      opts.optimizations.triangular_solve_expander_block_size = \
          triangular_solve_expander_block_size
      opts.optimizations.minimum_remote_tensor_size = \
          minimum_remote_tensor_size

      device_count = device_count_override or compute_device_count(
          pipelining, sharded, replicated)
      if device_count:
        opts.auto_select_ipus = device_count

      opts.serialization_output_folder = serialization_folder
      opts.ipu_model.compile_ipu_code = compile_ipu_code
      opts.ipu_model.tiles_per_ipu = tiles_per_ipu
      opts.allow_recompute = allow_recompute
      if num_io_tiles > 0:
        opts.io_tiles.num_io_tiles = num_io_tiles
        opts.io_tiles.place_ops_on_io_tiles = True

      opts.norms.use_stable_statistics = use_stable_norm_statistics

      if use_hw:
        add_hw_ci_connection_options(opts)

      if set_opts_fn:
        set_opts_fn(opts)

      if not estimator_hook:
        opts.configure_ipu_system()
      self.ipu_config = opts

  def create_ipu_event_trace(self):
    with ops.device('cpu'):
      self.report = gen_ipu_ops.ipu_event_trace()

  def reset(self):
    if self.eager_mode:
      self.create_ipu_event_trace()
    else:
      assert self.sess, "A valid session must be passed to the constructor" \
      " to use this method"
      self.sess.run(self.report)

  def parse_log(self, assert_len=None, assert_msg="", session=None):
    events = self.get_event_trace(session)
    return self.parse_events(events, assert_len, assert_msg)

  def get_ipu_events(self, session=None):
    events = self.get_event_trace(session)
    return self.get_events_from_log(events)

  def assert_no_event(self, msg=""):
    types, _ = self.get_ipu_events()
    self.test.assertFalse(types, msg)

  def assert_compiled_for_ipu(self, msg=""):
    types, _ = self.get_ipu_events()
    self.test.assertContainsSubset([IpuTraceEvent.COMPILE_END], types, msg)

  def get_event_trace(self, session=None):
    if self.eager_mode:
      assert session is None, "Sessions can't be used in eager mode"
      self.create_ipu_event_trace()
      events = self.report
    else:
      assert self.sess or session, "A valid session must be passed to either" \
      " the constructor or this method to be able to retrieve the log"
      assert self.report is not None, "No ipu_event_trace() has been created"

      if session:
        events = session.run(self.report)
      else:
        events = self.sess.run(self.report)
    return events

  def assert_num_events(self, num_expected, assert_msg=""):
    self.test.assertEqual(num_expected, self.num_events, assert_msg)

  def get_events_from_log(self, log):
    events_types = collections.defaultdict(int)
    events = []
    for e in log:
      if isinstance(e, ops.Tensor):
        e = e.numpy()
      assert isinstance(e, (bytes, str))
      evt = IpuTraceEvent.FromString(e)
      events_types[evt.type] += 1
      events.append(evt)
    return events_types, events

  def parse_events(self, events, assert_len=None, assert_msg=""):
    self.num_events = len(events)
    if assert_len:
      self.assert_num_events(assert_len, assert_msg)
    events_types, trace_events = self.get_events_from_log(events)
    self.tensor_map = None
    self.events = {}
    self.instruction_info = {}
    for evt in trace_events:
      try:
        if evt.type == IpuTraceEvent.COMPILE_BEGIN:
          pass
        if evt.type == IpuTraceEvent.COMPILE_END:
          if evt.compile_end.compilation_report:
            assert IpuTraceEvent.COMPILE_END not in self.events
            self.events[IpuTraceEvent.COMPILE_END] = js.loads(
                evt.compile_end.compilation_report, encoding="utf-8")
            self.tensor_map = TensorMap(
                js.loads(evt.compile_end.tensor_map, encoding="utf-8"),
                self.get_num_tiles_per_ipu())
            self.instruction_info = js.loads(evt.compile_end.instruction_info,
                                             encoding="utf-8")
        if evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
          if evt.data_transfer.data_transfer:
            assert IpuTraceEvent.HOST_TO_DEVICE_TRANSFER not in self.events
            self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER] = js.loads(
                evt.data_transfer.data_transfer, encoding="utf-8")
        if evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
          if evt.data_transfer.data_transfer:
            assert IpuTraceEvent.DEVICE_TO_HOST_TRANSFER not in self.events
            self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER] = js.loads(
                evt.data_transfer.data_transfer, encoding="utf-8")
        if evt.type == IpuTraceEvent.LOAD_ENGINE:
          pass
        if evt.type == IpuTraceEvent.EXECUTE:
          if evt.execute.execution_report:
            self.events[IpuTraceEvent.EXECUTE] = self.events.get(
                IpuTraceEvent.EXECUTE, []) + [
                    js.loads(evt.execute.execution_report, encoding="utf-8")
                ]
      except UnicodeDecodeError:
        pass
    return events_types

  def get_host_to_device_event_names(self):
    return [
        t["name"]
        for t in self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER]["tensors"]
    ]

  def get_device_to_host_event_names(self):
    return [
        t["name"]
        for t in self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]["tensors"]
    ]

  def assert_host_to_device_event_names(self, names, msg=None):
    self.test.assertEqual(
        len(names),
        len(
            self.events.get(IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                            {}).get("tensors", [])), msg)
    for name in names:
      self.test.assertEqual(
          count_matches_in_list(self.get_host_to_device_event_names(), name),
          1, msg)

  def assert_device_to_host_event_names(self, names, msg=None):
    self.test.assertEqual(
        len(names),
        len(
            self.events.get(IpuTraceEvent.DEVICE_TO_HOST_TRANSFER,
                            {}).get("tensors", [])), msg)
    for name in names:
      self.test.assertEqual(
          count_matches_in_list(self.get_device_to_host_event_names(), name),
          1, msg)

  def assert_num_execution_reports_equal(self, num):
    self.test.assertEqual(len(self.get_execution_reports()), num)

  def get_each_tile_memory(self):
    return self.events[IpuTraceEvent.COMPILE_END]["memory"]["byTile"]["total"]

  # Excluding gaps
  def get_max_tile_memory(self):
    return max(self.get_each_tile_memory())

  def get_always_live_memory(self):
    return sum(self.events[IpuTraceEvent.COMPILE_END]["memory"]["liveness"]
               ["alwaysLive"]["bytesByTile"])

  # Excluding always live
  def get_peak_liveness(self):
    # byProgram can be a nested tree of objects each with a "bytes" entry.
    def _helper(prog):
      # Reduce a generator so we only keep 2 values in memory at once.
      return reduce(max, map(_helper, prog["children"]), prog["bytes"])

    prog_livenesses = (self.events[IpuTraceEvent.COMPILE_END]["memory"]
                       ["liveness"]["notAlwaysLive"]["byProgram"])
    return reduce(max, map(_helper, prog_livenesses), 0)

  def get_total_tile_memory(self):
    return sum(self.get_each_tile_memory())

  def get_vertices(self):
    return self.events[IpuTraceEvent.COMPILE_END]["vertexTypes"]["names"]

  def get_compute_sets(self):
    return self.events[IpuTraceEvent.COMPILE_END]["computeSets"]["names"]

  def get_execution_reports(self):
    return self.events[IpuTraceEvent.EXECUTE]

  def get_execution_report_cycles(self, idx):
    return self.get_execution_reports()[idx]['simulation']['cycles']

  def get_instruction_info(self):
    return self.instruction_info

  def get_ml_type_counts(self):
    res = [0, 0, 0, 0]
    for i in self.instruction_info['ml_types'].values():
      ml_type = i - 1
      res[ml_type] = res[ml_type] + 1
    return res

  def assert_no_compute_set(self):
    self.test.assertFalse(
        self.events.get(IpuTraceEvent.COMPILE_END,
                        {}).get("computeSets", {}).get("names", {}))

  def assert_contains_one_compile_event(self):
    self.test.assertTrue(IpuTraceEvent.COMPILE_END in self.events)

  def assert_contains_no_compile_event(self):
    self.test.assertFalse(IpuTraceEvent.COMPILE_END in self.events)

  def assert_contains_host_to_device_transfer_event(self):
    assert IpuTraceEvent.HOST_TO_DEVICE_TRANSFER in self.events

  def assert_contains_device_to_host_transfer_event(self):
    assert IpuTraceEvent.DEVICE_TO_HOST_TRANSFER in self.events

  def get_tensor_map(self):
    return self.tensor_map

  def assert_tensor_inst_name_mappings(self, expected_mappings):
    mappings = self.get_tensor_map().tensor_inst_name_mappings()
    for inst, expected_name in expected_mappings.items():
      names = items_matching_at_least_one_pattern(mappings.keys(), [inst])
      self.test.assertTrue(
          names,
          "Couldn't find a match for '%s' in '%s'" % (inst, mappings.keys()))
      self.test.assertEqual(
          len(names), 1, "More than one match for '%s' : %s" % (inst, names))
      self.test.assertTrue(
          fnmatch.fnmatch(mappings[names[0]], expected_name),
          "Name '%s' for instruction '%s' does not match expected pattern '%s'"
          % (mappings[names[0]], names[0], expected_name))

  def assert_tensor_input_names(self, *expected_inputs):
    mappings = self.get_tensor_map().tensor_inst_name_mappings()
    for arg_num, expected_name in enumerate(expected_inputs):
      instrs = items_matching_at_least_one_pattern(mappings.keys(),
                                                   ["arg%d." % arg_num])
      assert len(instrs) == 1
      mangled = mappings[instrs[0]]
      m = re.match(r"XLA_Args/_arg_(.*)_0_0/_\d+", mangled)
      if not m:
        m = re.match("XLA_Args/(.*)", mangled)
      assert m
      self.test.assertTrue(
          fnmatch.fnmatch(m.group(1), expected_name),
          "Name '%s' for argument %d does not match expected pattern '%s'" %
          (m.group(1), arg_num, expected_name))

  def get_num_ipus(self):
    return self.events[IpuTraceEvent.COMPILE_END]["target"]["numIPUs"]

  def get_num_tiles(self):
    return self.events[IpuTraceEvent.COMPILE_END]["target"]["numTiles"]

  def get_num_tiles_per_ipu(self):
    return self.get_num_tiles() / self.get_num_ipus()

  def get_first_program_of_type(self, program_type):
    for p in self.events[IpuTraceEvent.COMPILE_END]["programs"]:
      if program_type == p['type']:
        return p
    return None

  def get_program_names_of_type(self, program_type):
    return [
        p['name'] for p in self.events[IpuTraceEvent.COMPILE_END]["programs"]
        if p['type'] == program_type
    ]

  def get_program(self, index=0):
    return self.events[IpuTraceEvent.COMPILE_END]["programs"][index]

  def assert_pipeline_stages_on_expected_ipu(self, expected_ipus):
    self.test.assertFalse(
        items_matching_at_least_one_pattern(
            self.tensor_map.computation_names(),
            ["*_stage_%d_" % (len(expected_ipus) + 1)]),
        "The number of expected_ipus does not match the number of stages")
    for i, expected_ipu in enumerate(expected_ipus):
      stage = items_matching_at_least_one_pattern(
          self.tensor_map.computation_names(), ["*_stage_%d_" % i])
      self.test.assertTrue(stage, "No stage %d found" % i)
      ipus = self.tensor_map.ipu_ids(stage)

      # A stage using device -1 can be on any of the devices.
      if ipus and expected_ipu != -1:
        self.test.assertEqual(
            len(ipus), 1,
            "Stage %d was mapped to more than one ipu: %s" % (i + 1, ipus))
        self.test.assertEqual(
            ipus.pop(), expected_ipu,
            "Stage %d did not run on the expected IPU" % (i + 1))

  def assert_each_tile_memory_is_less_than(self, expected, tolerance=0.01):
    low = 0
    high = int(expected * (1.0 + tolerance))
    self.test.assertAllInRange(self.get_each_tile_memory(), low, high)

  def assert_total_tile_memory(self, expected, tolerance=0.01):
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.test.assertAllInRange([self.get_total_tile_memory()], low, high)

  def assert_max_tile_memory(self, expected, tolerance=0.01):
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.test.assertAllInRange([self.get_max_tile_memory()], low, high)

  def assert_always_live_memory(self, expected, tolerance=0.01):
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.test.assertAllInRange([self.get_always_live_memory()], low, high)

  def assert_execution_report_cycles(self, idx, expected, tolerance=0.01):
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.test.assertAllInRange([self.get_execution_report_cycles(idx)], low,
                               high)

  # Asserts all the compute sets match a pattern in the whitelist and also asserts that all the whitelist patterns match at least one compute set
  def assert_all_compute_sets_and_list(self, ok):
    missing_whitelist = missing_whitelist_entries_in_names(
        self.get_compute_sets(), ok)
    missing_in_whitelist = missing_names_in_whitelist_entries(
        self.get_compute_sets(), ok)

    error_msg = "\n"
    if missing_whitelist:
      error_msg = "Whitelist items [%s] not found in compute sets:\n\t%s" % (
          ",".join(missing_whitelist), "\n\t".join(self.get_compute_sets()))
    if missing_whitelist and missing_in_whitelist:
      error_msg += "\n"
    if missing_in_whitelist:
      error_msg += "Compute sets items [%s] not found in whitelist:\n\t%s" % (
          ",".join(missing_in_whitelist), "\n\t".join(ok))

    self.test.assertFalse(missing_whitelist + missing_in_whitelist, error_msg)

  # Asserts all the global exchanges match a pattern in the whitelist and also asserts that all the whitelist patterns match at least one global exchange
  def assert_all_global_exchanges_and_list(self, ok):
    self.test.assertFalse(
        missing_whitelist_entries_in_names(
            self.get_program_names_of_type('GlobalExchange'),
            ok), "Whitelist items not found in global exchanges:\n\t%s" %
        "\n\t".join(self.get_compute_sets()))
    self.test.assertFalse(
        missing_names_in_whitelist_entries(
            self.get_program_names_of_type('GlobalExchange'),
            ok), "Global exchanges item not found in whitelist:\n\t%s" %
        "\n\t".join(ok))

  # Asserts that all the whitelist patterns match at least one compute set
  def assert_compute_sets_contain_list(self, ok):
    self.test.assertFalse(
        missing_whitelist_entries_in_names(self.get_compute_sets(), ok),
        "Whitelist items not found in compute sets:\n\t%s" %
        "\n\t".join(self.get_compute_sets()))

  # Asserts that none of the compute sets match any of the blacklist items
  def assert_compute_sets_not_in_blacklist(self, blacklist):
    self.test.assertFalse(
        names_in_blacklist(self.get_compute_sets(), blacklist),
        "Compute sets items found in blacklist:\n\t%s" %
        "\n\t".join(blacklist))

  # Asserts that all the whitelist patterns match at least one vertex
  def assert_vertices_contain_list(self, ok):
    self.test.assertFalse(
        missing_whitelist_entries_in_names(self.get_vertices(), ok),
        "Whitelist items not found in vertices:\n\t%s" %
        "\n\t".join(self.get_vertices()))

  def assert_compute_sets_matches(self, expr, num_matches, msg=None):
    self.test.assertEqual(count_matches_in_list(self.get_compute_sets(), expr),
                          num_matches, msg)


def count_ipu_compilations(events):
  count = 0
  for evt_str in events:
    evt = IpuTraceEvent.FromString(evt_str)
    if (evt.type == IpuTraceEvent.COMPILE_END
        and evt.compile_end.compilation_report):
      count += 1
  return count


def create_multi_increasing_dataset(value,
                                    shapes=None,
                                    dtypes=None,
                                    repeat=True):
  # Default values:
  shapes = shapes if shapes else [[1, 32, 32, 4], [1, 8]]
  dtypes = dtypes if dtypes else [np.float32, np.float32]

  def _get_one_input(data):
    result = []
    for i, shape in enumerate(shapes):
      result.append(
          math_ops.cast(gen_array_ops.broadcast_to(data, shape=shape),
                        dtype=dtypes[i]))
    return result

  dataset = Dataset.range(value).map(_get_one_input)
  if repeat:
    dataset = dataset.repeat()
  return dataset


def create_dual_increasing_dataset(value,
                                   data_shape=None,
                                   label_shape=None,
                                   dtype=np.float32,
                                   repeat=True):
  data_shape = data_shape if data_shape else [1, 32, 32, 4]
  label_shape = label_shape if label_shape else [1, 8]
  return create_multi_increasing_dataset(value,
                                         shapes=[data_shape, label_shape],
                                         dtypes=[dtype, dtype],
                                         repeat=repeat)


def create_single_increasing_dataset(value,
                                     shape=None,
                                     dtype=np.float32,
                                     repeat=True):
  shape = shape if shape is not None else [1, 32, 32, 4]
  return create_multi_increasing_dataset(value,
                                         shapes=[shape],
                                         dtypes=[dtype],
                                         repeat=repeat)


def move_variable_initialization_to_cpu():
  graph = ops.get_default_graph()

  init_ops = []
  dep_ops = [
      x.initializer.inputs[1].op for x in graph.get_collection('variables')
  ]
  visited = set()

  while dep_ops:
    op = dep_ops.pop()
    if not op in visited:
      visited.add(op)
      init_ops += [op]
      dep_ops += [x.op for x in op.inputs]

  # pylint: disable=protected-access
  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr(
        '_class',
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
            s=[b'loc:@cpu'])))
    op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
    op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))
  # pylint: enable=protected-access


def get_ci_num_ipus():
  return int(os.getenv('TF_IPU_COUNT', 0))


def has_ci_ipus():
  return get_ci_num_ipus() > 0


def add_hw_ci_connection_options(opts):
  opts.device_connection.enable_remote_buffers = True
  opts.device_connection.type = utils.DeviceConnectionType.ON_DEMAND


def test_may_use_ipus_or_model(num_ipus, func=None):
  """Test decorator for indicating that a test can run on both HW and Poplar
  IPU Model.
  Args:
  * num_ipus: number of IPUs required by the test.
  * func: the test function.
  """
  return test_uses_ipus(num_ipus=num_ipus, allow_ipu_model=True, func=func)


def test_uses_ipus(num_ipus, allow_ipu_model=False, func=None):
  """Test decorator for indicating how many IPUs the test requires. Allows us
  to skip tests which require too many IPUs.

  Args:
  * num_ipus: number of IPUs required by the test.
  * allow_ipu_model: whether the test supports IPUModel so that it can be
    executed without hardware.
  * func: the test function.
  """
  def decorator(f):
    def decorated(self, *args, **kwargs):
      num_available_ipus = get_ci_num_ipus()
      if num_available_ipus < num_ipus and not allow_ipu_model:
        self.skipTest(f"Requested {num_ipus} IPUs, but only "
                      f"{num_available_ipus} are available.")
      if num_available_ipus >= num_ipus:
        assert not ("use_ipu_model" in os.getenv(
            'TF_POPLAR_FLAGS',
            "")), "Do not set use_ipu_model when running HW tests."
      return f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def skip_on_hw(func):
  """Test decorator for skipping tests which should not be run on HW."""
  def decorator(f):
    def decorated(self, *args, **kwargs):
      if has_ci_ipus():
        self.skipTest("Skipping test on HW")

      return f(self, *args, **kwargs)

    return decorated

  return decorator(func)


def skip_with_asan(reason):
  """Test decorator for skipping tests which should not be run with AddressSanitizer."""
  if not isinstance(reason, str):
    raise TypeError("'reason' should be string, got {}".format(type(reason)))

  def decorator(f):
    def decorated(self, *args, **kwargs):
      if "ASAN_OPTIONS" in os.environ:
        self.skipTest(reason)

      return f(self, *args, **kwargs)

    return decorated

  return decorator


def skip_if_not_enough_ipus(self, num_ipus):
  num_available_ipus = get_ci_num_ipus()
  if num_available_ipus < num_ipus:
    self.skipTest(f"Requested {num_ipus} IPUs, but only "
                  f"{num_available_ipus} are available.")

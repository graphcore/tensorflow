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
# =============================================================================

import collections
import fnmatch
import os
import re
import json as js
import pathlib
import shutil
import tempfile

from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class ReportHelper():
  """ ReportHelper creates a temporary directory for reports to be generated
  in. `set_autoreport_options` configures poplar to use this directory.

  Reports are generated in unique subdirectories of the temporary directory and
  can be found by calling `find_report` or `find_reports`.

  Reports can also be cleared by calling `clear_reports`.

  All files are automatically cleaned up when the object is destroyed.
  """
  def __init__(self):
    self._directory = tempfile.mkdtemp(prefix="tf_")
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

  def set_autoreport_options(  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
      self,
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

  def find_reports(self):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Finds and returns the paths to generated report files in order of
    creation time (oldest first).
    """
    paths = []
    for d in self._find_report_subdirectories():
      files_ = list(self._find_report_files_in_subdirectory(d))
      # Only expect 1 report file per report subdirectory.
      if len(files_) != 1:
        raise IOError("Expected 1 report file in each report " +
                      f"subdirectory but found {len(files_)} in {d}:" +
                      "".join(f"\n   {f.name}" for f in files_))
      # Add report file absolute path to result.
      paths.append(str(files_[0]))

    # Sort by oldest first
    paths.sort(key=lambda p: os.stat(p).st_ctime)
    return paths

  def find_report(self):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Finds and returns the paths to the generated report file.
    Asserts the only one report has been generated.
    """
    reports = self.find_reports()
    num_reports = len(reports)
    assert num_reports == 1, f"Expected 1 report but found {num_reports}"
    return reports[0]

  def clear_reports(self):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Clears all existing reports and their subdirectories."""
    # Remove the whole directory and recreate it rather than removing each
    # subdirectory individually.
    shutil.rmtree(self._directory)
    os.mkdir(self._directory)

  # Automatically clean up all files when this instance is destroyed.
  def __del__(self):
    # Ignore errors to clean up as much as possible.
    shutil.rmtree(self._directory, ignore_errors=True)


class TensorMap:
  class Tile:
    def __init__(self, tile, num_elements):
      self.tile = tile
      self.num_elements = num_elements

    def __eq__(self, other):
      return self.tile == other.tile and self.num_elements == other.num_elements

  class Tensor:
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

  def __init__(self, tensor_map):
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

  def computation_names(self):
    return list(self.mappings.keys())

  def tensor_inst_name_mappings(self):
    mappings = {}
    for _, tensors in self.mappings.items():
      for tensor in tensors:
        assert tensor.id not in mappings, ("Instruction %s already in"
                                           " mappings %s") % (tensor.id,
                                                              mappings)
        mappings[tensor.id] = tensor.name
    return mappings


def _count_matches_in_list(input_list, to_match):
  return len([s for s in input_list if fnmatch.fnmatch(s, to_match)])


def _items_matching_at_least_one_pattern(items, patterns):
  matches = []
  patterns = [x + '*' for x in patterns]
  for item in items:
    if [p for p in patterns if fnmatch.fnmatch(item, p)]:
      matches.append(item)
  return matches


class ReportJSON:
  def __init__(self, test, sess=None, eager_mode=False):
    self.report = None
    self.test = test
    self.sess = sess
    self.eager_mode = eager_mode

    assert not eager_mode or not sess, "Sessions can't be used in eager mode"

    # If no session is passed to the constructor then assume
    # the events will be provided by the user.
    if sess:
      self.create_ipu_event_trace()

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
          if evt.compile_end.tensor_map:
            assert IpuTraceEvent.COMPILE_END not in self.events
            self.tensor_map = TensorMap(js.loads(evt.compile_end.tensor_map))
            self.instruction_info = js.loads(evt.compile_end.instruction_info)
        if evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
          if evt.data_transfer.data_transfer:
            # Save every transfer event
            if IpuTraceEvent.HOST_TO_DEVICE_TRANSFER not in self.events:
              self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER] = []
            data = js.loads(evt.data_transfer.data_transfer)
            self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER].append(data)
        if evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
          if evt.data_transfer.data_transfer:
            # Save every transfer event
            if IpuTraceEvent.DEVICE_TO_HOST_TRANSFER not in self.events:
              self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER] = []
            data = js.loads(evt.data_transfer.data_transfer)
            self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER].append(data)
        if evt.type == IpuTraceEvent.LOAD_ENGINE:
          pass
        if evt.type == IpuTraceEvent.EXECUTE:
          self.events[IpuTraceEvent.EXECUTE] = self.events.get(
              IpuTraceEvent.EXECUTE, [])
      except UnicodeDecodeError:
        pass
    return events_types

  def get_host_to_device_event_names(self, event_idx=0):
    return [
        t["name"] for t in self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER]
        [event_idx]["tensors"]
    ]

  def get_device_to_host_event_names(self, event_idx=0):
    return [
        t["name"] for t in self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]
        [event_idx]["tensors"]
    ]

  def assert_host_to_device_event_names(self, names, msg=None, event_idx=0):
    if names:
      self.test.assertTrue(
          IpuTraceEvent.HOST_TO_DEVICE_TRANSFER in self.events)
      transfer_events = self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER]
      self.test.assertTrue(len(transfer_events) > event_idx)
      event = transfer_events[event_idx]
      self.test.assertEqual(len(names), len(event.get('tensors', [])), msg)

      for name in names:
        self.test.assertEqual(
            _count_matches_in_list(
                self.get_host_to_device_event_names(event_idx=event_idx),
                name), 1, msg)

  def assert_device_to_host_event_names(self, names, msg=None, event_idx=0):
    if names:
      self.test.assertTrue(
          IpuTraceEvent.DEVICE_TO_HOST_TRANSFER in self.events)
      transfer_events = self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]
      self.test.assertTrue(len(transfer_events) > event_idx)
      event = transfer_events[event_idx]
      self.test.assertEqual(len(names), len(event.get('tensors', [])), msg)

      for name in names:
        self.test.assertEqual(
            _count_matches_in_list(
                self.get_device_to_host_event_names(event_idx=event_idx),
                name), 1, msg)

  def get_instruction_info(self):
    return self.instruction_info

  def get_ml_type_counts(self):
    res = [0, 0, 0, 0]
    for i in self.instruction_info['ml_types'].values():
      ml_type = i - 1
      res[ml_type] = res[ml_type] + 1
    return res

  def assert_contains_host_to_device_transfer_event(self):
    assert IpuTraceEvent.HOST_TO_DEVICE_TRANSFER in self.events

  def assert_contains_device_to_host_transfer_event(self):
    assert IpuTraceEvent.DEVICE_TO_HOST_TRANSFER in self.events

  def assert_num_host_to_device_transfer_events(self, num):
    assert len(self.events[IpuTraceEvent.HOST_TO_DEVICE_TRANSFER]) == num

  def assert_num_device_to_host_transfer_events(self, num):
    assert len(self.events[IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]) == num

  def get_tensor_map(self):
    return self.tensor_map

  def assert_tensor_inst_name_mappings(self, expected_mappings):
    mappings = self.get_tensor_map().tensor_inst_name_mappings()
    for inst, expected_name in expected_mappings.items():
      names = _items_matching_at_least_one_pattern(mappings.keys(), [inst])
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
      instrs = _items_matching_at_least_one_pattern(mappings.keys(),
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

  def assert_pipeline_stages_on_expected_ipu(self, expected_ipus,
                                             tile_per_ipu):
    self.test.assertFalse(
        _items_matching_at_least_one_pattern(
            self.tensor_map.computation_names(),
            ["*_stage_%d_" % (len(expected_ipus) + 1)]),
        "The number of expected_ipus does not match the number of stages")
    for i, expected_ipu in enumerate(expected_ipus):
      if not isinstance(expected_ipu, list) and expected_ipu != -1:
        stage = _items_matching_at_least_one_pattern(
            self.tensor_map.computation_names(), ["*_stage_%d_" % i])
        self.test.assertTrue(stage, "No stage %d found" % i)
        ipus = {
            tile // tile_per_ipu
            for tile in self.tensor_map.tile_ids(stage)
        }

        # A stage using device -1 can be on any of the devices.
        if ipus:
          self.test.assertEqual(
              len(ipus), 1,
              "Stage %d was mapped to more than one ipu: %s" % (i + 1, ipus))
          self.test.assertEqual(
              ipus.pop(), expected_ipu,
              "Stage %d did not run on the expected IPU" % (i + 1))


def missing_whitelist_entries_in_names(names, whitelist):
  fail_list = []
  wl = ['*' + x + '*' for x in whitelist]
  for x in wl:
    if not [name for name in names if fnmatch.fnmatch(name, x)]:
      fail_list += [x]
  return fail_list

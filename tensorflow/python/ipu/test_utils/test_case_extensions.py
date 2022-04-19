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

import fnmatch
from pva import ProgramVisitor

from tensorflow.python.framework.test_util import TensorFlowTestCase


def _count_matches_in_list(input_list, to_match):
  return len([s for s in input_list if fnmatch.fnmatch(s, to_match)])


# Members of this class are attached to TensorFlowTestCase.
class TestCaseExtensions:
  def _assert_all_in_tolerance(self, actual, expected, tolerance=0.001):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts that all values are within relative tolerance of expected.
    Only intended to be used with integer values.
    """
    low = int(expected * (1.0 - tolerance))
    high = int(expected * (1.0 + tolerance))
    self.assertAllInRange(actual, low, high)

  def assert_all_compute_sets_and_list(self, report, ok, ignore_common=True):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts all the compute sets match a pattern in the whitelist and also
    asserts that all the whitelist patterns match at least one compute set.
    """
    not_in_whitelist = []
    not_in_report = []
    whitelist = ['*' + x + '*' for x in ok]
    common_compute_sets = ['*__seed*', '*host-exchange-*', '*[cC]opy_*']

    for expected in whitelist:
      if not any(
          fnmatch.fnmatch(actual.name, expected)
          for actual in report.compilation.computeSets):
        not_in_report.append(expected)
    for actual in report.compilation.computeSets:
      if not any(
          fnmatch.fnmatch(actual.name, expected) for expected in whitelist):
        if ignore_common and any(
            fnmatch.fnmatch(actual.name, c) for c in common_compute_sets):
          continue

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

  def assert_num_reports(self, report_helper, n):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts the number of reports found by the given report helper."""
    self.assertLen(report_helper.find_reports(), n)

  def assert_compute_sets_matches(self, report, expr, num_matches, msg=None):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts the number of compute sets in the report which match expr."""
    cs_names = (cs.name for cs in report.compilation.computeSets)
    self.assertEqual(_count_matches_in_list(cs_names, expr), num_matches, msg)

  def assert_compute_sets_contain_list(self, report, ok):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
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

  def assert_compute_sets_not_in_blacklist(self, report, blacklist):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
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

  def assert_vertices_contain_list(self, report, ok):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts that all the whitelist patterns match at least one vertex."""
    whitelist = ['*' + x + '*' for x in ok]
    not_in_report = []

    def vertex_iterator():
      for cs in report.compilation.computeSets:
        for v in cs.vertices:
          yield v

    for expected in whitelist:
      if not any(
          fnmatch.fnmatch(v.type.name, expected) for v in vertex_iterator()):
        not_in_report.append(expected)

    self.assertFalse(
        not_in_report, "Whitelist items [%s] not found in vertices:\n\t%s" %
        (",".join(not_in_report), "\n\t".join(v.type.name
                                              for v in vertex_iterator())))

  def assert_each_tile_memory_is_less_than(  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
      self,
      report,
      max_expected,
      tolerance=0.01):
    """Assert total memory (excluding gaps) on each tile is below max."""
    high = int(max_expected * (1.0 + tolerance))
    tile_memory = [
        tile.memory.total.excludingGaps for tile in report.compilation.tiles
    ]
    self.assertAllInRange(tile_memory, 0, high)

  def assert_total_tile_memory(self, report, expected, tolerance=0.01):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Assert total memory (excluding gaps) across all tiles is close to
    expected.
    """
    total_memory = sum(tile.memory.total.excludingGaps
                       for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([total_memory], expected, tolerance)

  def assert_max_tile_memory(self, report, expected, tolerance=0.01):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Assert peak tile memory (excluding gaps) is close to expected."""
    max_memory = max(tile.memory.total.excludingGaps
                     for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([max_memory], expected, tolerance)

  def assert_always_live_memory(self, report, expected, tolerance=0.01):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Assert total always-live memory across all tiles is close to
    expected.
    """
    always_live_memory = sum(tile.memory.alwaysLiveBytes
                             for tile in report.compilation.tiles)
    self._assert_all_in_tolerance([always_live_memory], expected, tolerance)

  def assert_execution_report_cycles(  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
      self,
      report,
      expected,
      tolerance=0.01,
      expected_executions=1):
    """Asserts the total cycles on each tile are close to expected for the
    specified execution.
    """
    self.assert_number_of_executions(report, expected_executions)
    cycles = [report.execution.totalCycles.total]
    self._assert_all_in_tolerance(cycles, expected, tolerance)

  def assert_number_of_executions(self, report, n):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts the number of executions in the report."""
    self.assertLen(report.execution.runs, n)

  def assert_compute_io_overlap_percentage(self, report, p):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts a minimum percentage of compute and IO overlapped in the execution
    report.
    """
    # Execution steps for the first run
    steps = report.execution.runs[0].steps
    computeIntervals = []
    streamCopyIntervals = []

    class IntervalVisitor(ProgramVisitor):
      def __init__(self, cyclesFrom, cyclesTo):
        self.cyclesFrom = cyclesFrom
        self.cyclesTo = cyclesTo
        super().__init__()

      def visitOnTileExecute(self, _):
        computeIntervals.append([self.cyclesFrom.max, self.cyclesTo.max])

      def visitStreamCopyMid(self, _):
        streamCopyIntervals.append([self.cyclesFrom.max, self.cyclesTo.max])

    for step in steps:
      ipu = step.ipus[0]
      f = ipu.activeCycles.cyclesFrom
      t = ipu.activeCycles.cyclesTo
      v = IntervalVisitor(f, t)
      step.program.accept(v)

      def checkOverlap(low1, high1, low2, high2):
        return low1 < high2 and low2 < high1

      def getOverlap(low1, high1, low2, high2):
        overlap = min(high1, high2) - max(low1, low2)
        return overlap

      overlap = 0
      for compute in computeIntervals:
        for stream in streamCopyIntervals:
          if checkOverlap(compute[0], compute[1], stream[0], stream[1]):
            overlap += getOverlap(compute[0], compute[1], stream[0], stream[1])

    computeTotal, streamCopyTotal = (
        sum(i[1] - i[0] for i in intervals)
        for intervals in [computeIntervals, streamCopyIntervals])

    self.assertGreater(max(overlap / streamCopyTotal, overlap / computeTotal),
                       p)

  def assert_global_exchange_percentage(self, report, p):  # pylint: disable=missing-param-doc,missing-type-doc,missing-return-doc,missing-return-type-doc
    """Asserts a maximum percentage of global exchange in the execution report.
    """
    class IntervalVisitor(ProgramVisitor):
      def __init__(self):
        self.globalExchangeCycles = 0
        super().__init__()

      def visitGlobalExchange(self, globalExchange):
        self.globalExchangeCycles += (globalExchange.cyclesTo.max -
                                      globalExchange.cyclesFrom.max)

    v = IntervalVisitor()
    totalCycles = 0
    for step in report.execution.runs[0].steps:
      ipu = step.ipus[0]
      totalCycles += (ipu.activeCycles.cyclesTo.max -
                      ipu.activeCycles.cyclesFrom.max)
      step.program.accept(v)

    self.assertGreaterEqual(p, v.globalExchangeCycles / totalCycles)


# Attach everything from TestCaseExtensions onto TensorFlowTestCase.
# Note that XLATestCase inherits from TensorFlowTestCase.
for attr in dir(TestCaseExtensions):
  if not attr.startswith('__'):
    setattr(TensorFlowTestCase, attr, getattr(TestCaseExtensions, attr))

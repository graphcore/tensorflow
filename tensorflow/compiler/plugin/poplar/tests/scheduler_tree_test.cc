/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <set>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/schedule_tree.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

struct ForEachPredecessor {
  ForEachPredecessor(std::map<char, std::set<char>> predecessors)
      : predecessors_(predecessors) {}

  template <typename F>
  void operator()(char node, F f) const {
    if (predecessors_.count(node) > 0) {
      for (auto c : predecessors_.at(node)) {
        f(c);
      }
    }
  }

 private:
  std::map<char, std::set<char>> predecessors_;
};

struct ForEachSucessor {
  ForEachSucessor(std::map<char, std::set<char>> successors)
      : successors_(successors) {}

  template <typename F>
  void operator()(char node, F f) const {
    if (successors_.count(node) > 0) {
      for (auto c : successors_.at(node)) {
        f(c);
      }
    }
  }

 private:
  std::map<char, std::set<char>> successors_;
};

std::map<char, std::set<char>> createPredecessorsFromSucessors(
    std::map<char, std::set<char>> successors) {
  std::map<char, std::set<char>> predecessors;

  for (const auto& successor_pair : successors) {
    for (auto successor : successor_pair.second) {
      predecessors[successor].insert(successor_pair.first);
    }
  }

  return predecessors;
}

struct CharGrossCost {
  explicit CharGrossCost(std::map<char, int64_t> gross_cost)
      : gross_cost_(gross_cost) {}

  int64_t operator()(char node) const {
    if (gross_cost_.count(node) > 0) {
      return gross_cost_.at(node);
    }

    return 0;
  }

 private:
  std::map<char, int64_t> gross_cost_;
};

struct CharTempCost {
  explicit CharTempCost(std::map<char, int64_t> temp_cost)
      : temp_cost_(temp_cost) {}

  template <typename Set>
  int64_t operator()(const Set&, char node) const {
    if (temp_cost_.count(node) > 0) {
      return temp_cost_.at(node);
    }

    return 0;
  }

 private:
  std::map<char, int64_t> temp_cost_;
};

TEST(SchedulerTreeTest, SimpleSchedule) {
  using CharScheduleTree =
      ScheduleTree<char, ForEachPredecessor, ForEachSucessor, CharGrossCost,
                   CharTempCost>;
  std::vector<char> instructions('G' - 'A');
  std::iota(instructions.begin(), instructions.end(), 'A');

  // clang-format off
  std::map<char, std::set<char>> successors = {
    {'A', {'B', 'C'}},
    {'B', {'D'}},
    {'C', {'E'}},
    {'D', {'F'}},
    {'E', {'F'}},
  };
  std::map<char, int64_t> gross_cost = {
      {'A', 2},
      {'B', 1},
      {'C', 4},
      {'D', 4},
      {'E', 1},
      {'F', 0},
  };
  // clang-format on

  std::map<char, int64_t> temp_cost = {};
  std::map<char, std::set<char>> predecessors =
      createPredecessorsFromSucessors(successors);

  auto tree = std::make_shared<const CharScheduleTree>(
      instructions, ForEachPredecessor(predecessors),
      ForEachSucessor(successors), CharGrossCost(gross_cost),
      CharTempCost(temp_cost));

  tree = tree->Grow(3);
  while (!tree->IsLeaf()) {
    tree = tree->BestChild()->Grow();
  }

  auto schedule = tree->GetSchedule();
  std::vector<char> expected_schedule = {'A', 'B', 'C', 'E', 'D', 'F'};

  ASSERT_EQ(instructions.size(), schedule.size())
      << "Didn't schedule all operations";
  for (int i = 0; i < instructions.size(); ++i) {
    EXPECT_EQ(expected_schedule[i], schedule[i])
        << "Vectors expected_schedule and schedule differ at index " << i;
  }
}

TEST(SchedulerTreeTest, MultiBranchMergeSchedule) {
  using CharScheduleTree =
      ScheduleTree<char, ForEachPredecessor, ForEachSucessor, CharGrossCost,
                   CharTempCost>;
  std::vector<char> instructions('K' - 'A');
  std::iota(instructions.begin(), instructions.end(), 'A');

  // clang-format off
  std::map<char, std::set<char>> successors = {
    {'A', {'B', 'C'}},
    {'B', {'D'}},
    {'C', {'E'}},
    {'D', {'F'}},
    {'E', {'F'}},
    {'F', {'G', 'H'}},
    {'G', {'J'}},
    {'H', {'I'}},
    {'I', {'J'}},
  };
  std::map<char, int64_t> gross_cost = {
      {'A', 2},
      {'B', 1},
      {'C', 4},
      {'D', 4},
      {'E', 1},
      {'F', 2},
      {'G', 3},
      {'H', 10},
      {'I', 1},
      {'J', 0},
  };
  // clang-format on

  std::map<char, int64_t> temp_cost = {};
  std::map<char, std::set<char>> predecessors =
      createPredecessorsFromSucessors(successors);

  auto tree = std::make_shared<const CharScheduleTree>(
      instructions, ForEachPredecessor(predecessors),
      ForEachSucessor(successors), CharGrossCost(gross_cost),
      CharTempCost(temp_cost));

  // Lookahead
  tree = tree->Grow(3);
  while (!tree->IsLeaf()) {
    tree = tree->BestChild()->Grow();
    // VLOG(0) << "Scheduling " << tree->GetSchedule().back();
    // VLOG(0) << "Max Liveness " << tree->MaxLiveness();
  }

  auto schedule = tree->GetSchedule();
  std::vector<char> expected_schedule = {'A', 'B', 'C', 'E', 'D',
                                         'F', 'H', 'I', 'G', 'J'};

  ASSERT_EQ(instructions.size(), schedule.size())
      << "Didn't schedule all operations";
  for (int i = 0; i < instructions.size(); ++i) {
    EXPECT_EQ(expected_schedule[i], schedule[i])
        << "Vectors expected_schedule and schedule differ at index " << i;
  }
}

TEST(SchedulerTreeTest, MultiBranchSchedule) {
  using CharScheduleTree =
      ScheduleTree<char, ForEachPredecessor, ForEachSucessor, CharGrossCost,
                   CharTempCost>;
  std::vector<char> instructions('K' - 'A');
  std::iota(instructions.begin(), instructions.end(), 'A');

  // clang-format off
  std::map<char, std::set<char>> successors = {
    {'A', {'B', 'C'}},
    {'B', {'D'}},
    {'C', {'E'}},
    {'D', {'F'}},
    {'E', {'F'}},
    {'F', {'G', 'H'}},
    {'G', {'J'}},
    {'H', {'I'}},
  };
  std::map<char, int64_t> gross_cost = {
      {'A', 2},
      {'B', 1},
      {'C', 4},
      {'D', 4},
      {'E', 1},
      {'F', 2},
      {'G', 3},
      {'H', 10},
      {'I', 0},
      {'J', 3},
  };
  // clang-format on

  std::map<char, int64_t> temp_cost = {};
  std::map<char, std::set<char>> predecessors =
      createPredecessorsFromSucessors(successors);

  auto tree = std::make_shared<const CharScheduleTree>(
      instructions, ForEachPredecessor(predecessors),
      ForEachSucessor(successors), CharGrossCost(gross_cost),
      CharTempCost(temp_cost));

  tree = tree->Grow(3);
  while (!tree->IsLeaf()) {
    tree = tree->BestChild()->Grow();
  }

  auto schedule = tree->GetSchedule();
  std::vector<char> expected_schedule = {'A', 'B', 'C', 'E', 'D',
                                         'F', 'H', 'I', 'G', 'J'};

  ASSERT_EQ(instructions.size(), schedule.size())
      << "Didn't schedule all operations";
  for (int i = 0; i < instructions.size(); ++i) {
    EXPECT_EQ(expected_schedule[i], schedule[i])
        << "Vectors expected_schedule and schedule differ at index " << i;
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

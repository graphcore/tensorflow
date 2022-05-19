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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SCHEDULE_TREE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SCHEDULE_TREE_H_

#include <limits>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace poplarplugin {

/**
 * The schedule search tree class.
 *
 * This class helps to find the minimum max-liveness schudule.
 *
 * @tparam ElemType The schedule element type.
 * @tparam ElemPreVisitor The schedule element predecessor visitor wrapper type.
 *                        This must be a callable type with signature
 *                        `void(ElemType, void(ElemType))`.
 * @tparam ElemPostVisitor The schedule element successor visitor wrapper type.
 *                         This must be a callable type with signature
 *                         `void(ElemType, void(ElemType))`.
 * @tparam GrossCostFunc The schedule element gross cost callable type.
 *                       This must be callable with a signature
 *                       `int64_t(ElemType)`.
 * @tparam TempCostFunc The schedule element temporary cost callable type.
 *                      This must be callable with a signature
 *                      `int64_t(Set<ElemType> live, ElemType)`.
 * @tparam ElemCompFunc The comparison type for ordering ElemType.
 */
template <typename ElemType, typename ElemPreVisitor, typename ElemPostVisitor,
          typename GrossCostFunc, typename TempCostFunc,
          typename ElemCompFunc = std::less<ElemType>>
class ScheduleTree
    : public std::enable_shared_from_this<
          ScheduleTree<ElemType, ElemPreVisitor, ElemPostVisitor, GrossCostFunc,
                       TempCostFunc, ElemCompFunc>> {
 public:
  using ThisType = ScheduleTree<ElemType, ElemPreVisitor, ElemPostVisitor,
                                GrossCostFunc, TempCostFunc, ElemCompFunc>;
  using ThisTypePtr = std::shared_ptr<ThisType const>;
  using InstructionCostMap = absl::flat_hash_map<ElemType, int64_t>;
  using InstructionWaitList =
      absl::flat_hash_map<int64_t, std::set<ElemType, ElemCompFunc>>;
  using InverseInstructionWaitList = absl::flat_hash_map<ElemType, int64_t>;

  ScheduleTree(const std::vector<ElemType>& elems,
               ElemPreVisitor elem_pre_visit = {},
               ElemPostVisitor elem_post_visit = {}, GrossCostFunc cost_f = {},
               TempCostFunc temp_cost_f = {})
      : ScheduleTree(cost_f, temp_cost_f, elem_pre_visit, elem_post_visit) {
    for (auto& elem : elems) {
      auto count = GetOperandCount(elem);
      wait_list_[count].insert(elem);
      inv_wait_list_[elem] = count;
    }
  }

  /**
   * Return the child node with the lowest max liveness.
   *
   * @param search_limit Limit the total number of nodes to search.
   *
   * @returns The node which has the minimum max-liveness within the search
   *          horizon.
   */
  ThisTypePtr BestChild(
      int64_t search_limit = std::numeric_limits<int64_t>::max()) const {
    ThisTypePtr best = nullptr;
    int64_t best_max_liveness = std::numeric_limits<int64_t>::max();

    if (children_.size() == 1) {
      return *children_.begin();
    }

    for (auto& child : children_) {
      auto child_max_liveness = child->MaxLiveness(
          best_max_liveness, search_limit / children_.size());

      if (child_max_liveness < best_max_liveness) {
        best = child;
        best_max_liveness = child_max_liveness;
      }
    }

    return best;
  }

  /**
   * Compute the current liveness, including temporary costs.
   *
   * @returns The current liveness in bytes.
   */
  int64_t CurrentLiveness() const {
    if (current_liveness_ == std::numeric_limits<int64_t>::max()) {
      current_liveness_ =
          temp_cost_f_(GetCurrentlyLive(), schedule_prefix_.back());
      for (auto pair : inv_live_list_) {
        current_liveness_ += cost_f_(pair.first);
      }
    }

    return current_liveness_;
  }

  /**
   * Compute the max liveness of this node an its best child.
   *
   * @param max_liveness The current max-liveness. This is used for early
   *                     temrination.
   * @param search_limit This is an upper-bound on the number of nodes to visit.
   *
   * @returns The minimum max-liveness found from this this node.
   */
  int64_t MaxLiveness(
      int64_t max_liveness = std::numeric_limits<int64_t>::max(),
      int64_t search_limit = std::numeric_limits<int64_t>::max()) const {
    auto current_liveness = CurrentLiveness();

    if ((children_.empty()) || (current_liveness >= max_liveness) ||
        (search_limit <= 0)) {
      return current_liveness;
    }

    int64_t best_max_liveness =
        std::min(max_liveness, std::numeric_limits<int64_t>::max());
    search_limit = (search_limit - 1) / children_.size();
    for (auto& child : children_) {
      best_max_liveness =
          std::min(best_max_liveness,
                   child->MaxLiveness(best_max_liveness, search_limit));
    }

    return std::max(current_liveness, best_max_liveness);
  }

  /**
   * Get the schedule that ends at this node.
   *
   * If this node is a leaf, then this returns a valid schedule.
   *
   * @returns The schedule that ends at this node.
   */
  const std::vector<ElemType>& GetSchedule() const { return schedule_prefix_; }

  /**
   * Get the set of currently live elements.
   *   *
   * @returns The set of live elements.
   */
  std::set<ElemType, ElemCompFunc> GetCurrentlyLive() const {
    std::set<ElemType, ElemCompFunc> result;

    for (auto pair : live_list_) {
      if (pair.first > 0) {
        result.insert(pair.second.begin(), pair.second.end());
      }
    }

    return result;
  }

  /**
   * Get the set of elements that are ready to be scheduled.
   *
   * @returns The set of ready elements
   */
  const std::set<ElemType, ElemCompFunc>& GetReady() const {
    return wait_list_.at(0);
  }

  /**
   * Reduce the waiting count of a given element.
   *
   * @param elem The element to reduce the waiting count for.
   */
  void ReduceWaiting(ElemType elem) {
    auto& count = inv_wait_list_.at(elem);
    wait_list_[count].erase(elem);
    count--;
    wait_list_[count].insert(elem);
    inv_wait_list_[elem] = count;
  }

  /**
   * Reduce the live count of a given element.
   *
   * @param elem The element to reduce the live count for.
   */
  void ReduceLive(ElemType elem) {
    auto& count = inv_live_list_.at(elem);
    live_list_[count].erase(elem);
    count--;
    if (count > 0) {
      live_list_[count].insert(elem);
    } else {
      inv_live_list_.erase(elem);
    }
  }

  /**
   * Schedule the given element.
   *
   * @param elem The elements to be scheduled.
   */
  void Schedule(ElemType elem) {
    auto use_count = GetUserCount(elem);

    live_list_[use_count].insert(elem);
    inv_live_list_[elem] = use_count;

    schedule_prefix_.push_back(elem);
    wait_list_.at(0).erase(elem);
  }

  /**
   * Grow the leaves of the tree.
   *
   * @param depth The amount to grow the leaf nodes.
   * @param search_limit This is an upper-bound on the number of nodes to visit.
   *
   * @returns The new tree node, which has been grown.
   */
  ThisTypePtr Grow(
      int depth = 1,
      int64_t search_limit = std::numeric_limits<int64_t>::max()) const {
    if (depth <= 0 || search_limit <= 1) {
      return this->shared_from_this();
    }

    if (!IsLeaf() && (wait_list_.at(0).size() == children_.size())) {
      auto result = std::make_shared<ThisType>(*this);

      for (auto& child : result->children_) {
        child = child->Grow(depth, (search_limit - 1) / children_.size());
      }

      return result;
    }

    if (!IsLeaf() && children_.empty()) {
      auto result = std::make_shared<ThisType>(*this);

      for (auto elem : result->GetReady()) {
        auto child = std::make_shared<ThisType>(*this);
        child->current_liveness_ = std::numeric_limits<int64_t>::max();

        child->Schedule(elem);

        elem_post_visit_(
            elem, [&](ElemType successor) { child->ReduceWaiting(successor); });

        elem_pre_visit_(elem, [&](ElemType predecessor) mutable {
          child->ReduceLive(predecessor);
        });

        result->children_.emplace_back(std::move(child));
      }

      return result->Grow(depth - 1, search_limit);
    }

    return this->shared_from_this();
  }

  /**
   * Take all the ready nodes that satisfy the given predicate.
   *
   * @param predicate A unary predicate.
   *
   * @returns The new tree node, which has had the set of ready elements
   *          scheduled.
   */
  template <typename UnaryPredicateType>
  ThisTypePtr TakeAllReady(UnaryPredicateType predicate) const {
    auto result = std::make_shared<ThisType>(*this);
    result->current_liveness_ = std::numeric_limits<int64_t>::max();

    auto ready = GetReady();
    for (auto elem : ready) {
      if (predicate(elem)) {
        result->Schedule(elem);

        elem_post_visit_(elem, [&](ElemType successor) {
          result->ReduceWaiting(successor);
        });

        elem_pre_visit_(elem, [&](ElemType predecessor) mutable {
          result->ReduceLive(predecessor);
        });
      }
    }

    return result;
  }

  /**
   * Take all the ready nodes.
   *
   * @returns The new tree node, which has had the set of ready elements
   *          scheduled.
   */
  ThisTypePtr TakeAllReady() const {
    return TakeAllReady([](ElemType) { return true; });
  }

  /**
   * Check if a node is a leaf.
   *
   * @returns Whether the current node is a leaf node.
   */
  bool IsLeaf() const { return wait_list_.at(0).empty(); }

 private:
  ScheduleTree(GrossCostFunc cost_f, TempCostFunc temp_cost_f,
               ElemPreVisitor elem_pre_visit, ElemPostVisitor elem_post_visit)
      : cost_f_(cost_f),
        temp_cost_f_(temp_cost_f),
        elem_pre_visit_(elem_pre_visit),
        elem_post_visit_(elem_post_visit) {}

  int64_t GetOperandCount(ElemType elem) const {
    int64_t result = 0;

    elem_pre_visit_(elem, [&](ElemType) mutable { result++; });

    return result;
  }

  int64_t GetUserCount(ElemType elem) const {
    int64_t result = 0;

    elem_post_visit_(elem, [&](ElemType) mutable { result++; });

    return result;
  }

  GrossCostFunc cost_f_;
  TempCostFunc temp_cost_f_;
  ElemPreVisitor elem_pre_visit_;
  ElemPostVisitor elem_post_visit_;
  std::vector<ThisTypePtr> children_;
  std::vector<ElemType> schedule_prefix_;
  InstructionWaitList wait_list_;
  InverseInstructionWaitList inv_wait_list_;
  InstructionWaitList live_list_;
  InverseInstructionWaitList inv_live_list_;
  mutable int64_t current_liveness_ = std::numeric_limits<int64_t>::max();
};

}  // namespace poplarplugin
}  // namespace xla

#endif

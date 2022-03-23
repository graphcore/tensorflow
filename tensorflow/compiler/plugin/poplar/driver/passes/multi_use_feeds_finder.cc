/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_use_feeds_finder.h"

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {
using PredFn = std::function<bool(const HloInstruction*)>;
using GetFeedConfigFn = std::function<PoplarFeedConfig(const HloInstruction*)>;
using SetFeedConfigFn =
    std::function<void(HloInstruction*, const std::string&)>;

StatusOr<bool> HandleFeeds(HloModule* module, PredFn predicate,
                           GetFeedConfigFn get_config,
                           SetFeedConfigFn set_config) {
  bool changed = false;

  absl::flat_hash_map<std::string, std::vector<HloInstruction*>> feeds;

  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* inst : comp->instructions()) {
      if (predicate(inst)) {
        PoplarFeedConfig config = get_config(inst);
        feeds[config.feed_id()].push_back(inst);
      }
    }
  }

  for (auto pair : feeds) {
    auto& feed_id = pair.first;
    auto& insts = pair.second;
    if (insts.size() == 1) {
      continue;
    }
    VLOG(3) << "Found feed id " << feed_id << " being used multiple times.";
    // Check all the sharding information matches.
    HloInstruction* first_inst = insts[0];
    const bool all_sharding_same =
        absl::c_all_of(insts, [&first_inst](const HloInstruction* inst) {
          return first_inst->has_compatible_sharding(inst);
        });
    if (!all_sharding_same) {
      return UnimplementedStrCat(
          "Trying to use the same feed with id", feed_id,
          " on different shards which is not supported.");
    }
    // Mark all the feeds as reusable.
    for (HloInstruction* inst : insts) {
      VLOG(2) << "Marking feed " << inst->ToString() << " as reusable.";
      PoplarFeedConfig config = get_config(inst);
      config.set_reusable(true);
      std::string config_str;
      if (!config.SerializeToString(&config_str)) {
        return InternalError("Could not serialize feed config");
      }
      set_config(inst, config_str);
    }
    changed = true;
  }
  return changed;
}
}  // namespace

StatusOr<bool> MultiUseFeedsFinder::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(
      bool changed_infeeds,
      HandleFeeds(module,
                  [](const HloInstruction* inst) {
                    return inst->opcode() == HloOpcode::kInfeed;
                  },
                  [](const HloInstruction* inst) {
                    PoplarFeedConfig config;
                    config.ParseFromString(inst->infeed_config());
                    return config;
                  },
                  [](HloInstruction* inst, const std::string& config_str) {
                    inst->set_infeed_config(config_str);
                  }));

  TF_ASSIGN_OR_RETURN(
      bool changed_outfeeds,
      HandleFeeds(module,
                  [](const HloInstruction* inst) {
                    return inst->opcode() == HloOpcode::kOutfeed;
                  },
                  [](const HloInstruction* inst) {
                    PoplarFeedConfig config;
                    config.ParseFromString(inst->outfeed_config());
                    return config;
                  },
                  [](HloInstruction* inst, const std::string& config_str) {
                    inst->set_outfeed_config(config_str);
                  }));

  return changed_infeeds || changed_outfeeds;
}

}  // namespace poplarplugin
}  // namespace xla

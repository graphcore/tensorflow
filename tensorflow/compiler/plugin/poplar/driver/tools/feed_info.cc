/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/feed_info.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {
PoplarFeedConfig GetFeedConfig(const HloInstruction* inst) {
  PoplarFeedConfig config;
  if (inst->opcode() == HloOpcode::kInfeed) {
    config.ParseFromString(inst->infeed_config());
  } else {
    CHECK_EQ(inst->opcode(), HloOpcode::kOutfeed);
    config.ParseFromString(inst->outfeed_config());
  }
  return config;
}

Status SetFeedConfig(HloInstruction* inst, PoplarFeedConfig& config) {
  std::string config_str;
  if (!config.SerializeToString(&config_str)) {
    return InternalError("Could not serialize feed config");
  }
  if (inst->opcode() == HloOpcode::kInfeed) {
    inst->set_infeed_config(config_str);
  } else {
    CHECK_EQ(inst->opcode(), HloOpcode::kOutfeed);
    inst->set_outfeed_config(config_str);
  }
  return Status::OK();
}

Status HandleFeed(HloInstruction* inst, uint64& next_feed_id,
                  FeedNameMapping& original_to_canonical_name,
                  FeedNameMapping& canonical_to_original_name) {
  PoplarFeedConfig config = GetFeedConfig(inst);
  const std::string feed_id = config.feed_id();
  if (!original_to_canonical_name.contains(feed_id)) {
    const std::string canonical_feed_id = std::to_string(++next_feed_id);
    original_to_canonical_name[feed_id] = canonical_feed_id;
    canonical_to_original_name[canonical_feed_id] = feed_id;
  }
  config.set_feed_id(original_to_canonical_name.at(feed_id));
  VLOG(2) << "Mapping infeed " << feed_id << " to canonical name "
          << config.feed_id();
  return SetFeedConfig(inst, config);
}
}  // namespace

StatusOr<FeedNameMappings> CanonicalizeFeedsInModule(HloModule* module) {
  VLOG(2) << "Canonicalizing feeds for " << module->name();
  FeedNameMapping infeed_original_to_canonical_name,
      infeed_canonical_to_original_name;
  FeedNameMapping outfeed_original_to_canonical_name,
      outfeed_canonical_to_original_name;

  uint64 next_feed_id = 0;
  // Replace the infeed and outfeed names in a deterministic order.
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kInfeed) {
        TF_RETURN_IF_ERROR(HandleFeed(inst, next_feed_id,
                                      infeed_original_to_canonical_name,
                                      infeed_canonical_to_original_name));
      } else if (inst->opcode() == HloOpcode::kOutfeed) {
        TF_RETURN_IF_ERROR(HandleFeed(inst, next_feed_id,
                                      outfeed_original_to_canonical_name,
                                      outfeed_canonical_to_original_name));
      }
    }
  }

  return FeedNameMappings{infeed_canonical_to_original_name,
                          outfeed_canonical_to_original_name};
}
}  // namespace poplarplugin
}  // namespace xla

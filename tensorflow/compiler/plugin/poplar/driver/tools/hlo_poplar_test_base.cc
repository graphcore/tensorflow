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

#include <utility>

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {

poplar::Device HloPoplarTestBase::CreateIpuModel(int32 num_ipus,
                                                 int32 num_tiles) {
  poplar::IPUModel model;
  if (num_ipus) {
    model.numIPUs = num_ipus;
  }
  if (num_tiles) {
    model.tilesPerIPU = num_tiles;
  }
  return model.createDevice();
}

StatusOr<poplar::Device> HloPoplarTestBase::CreateIpuDevice(int32 num_ipus,
                                                            int32 num_tiles) {
  poplar::DeviceManager& manager = PoplarExecutor::GetDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, num_ipus);
  TF_ASSIGN_OR_RETURN(std::size_t device_index,
                      PoplarExecutor::AttachToPoplarDevice(devices, 0, true));
  auto device = std::move(devices.at(device_index));

  if (num_tiles <= 0) {
    num_tiles = device.getTarget().getTilesPerIPU();
  }

  if (num_tiles != device.getTarget().getTilesPerIPU()) {
    return device.createVirtualDevice(num_tiles);
  } else {
    return device;
  }
}

std::unique_ptr<CompilerResources> HloPoplarTestBase::GetMockResources(
    HloModule* module) {
  auto device = poplar::Device::createCPUDevice();
  return GetMockResources(device, module);
}

std::unique_ptr<CompilerResources> HloPoplarTestBase::GetMockResources(
    HloModule* module, bool merge_infeeds) {
  auto device = poplar::Device::createCPUDevice();
  auto resources = GetMockResources(device, module);
  resources->merge_infeed_io_copies = merge_infeeds;
  return std::move(resources);
}

std::unique_ptr<CompilerResources> HloPoplarTestBase::GetMockResources(
    poplar::Device& device, HloModule* module, bool merge_infeeds,
    int number_of_vgraphs, int64 max_inter_ipu_copies_buffer_size) {
  const auto info = CompilerInformation().set_max_inter_ipu_copies_buffer_size(
      max_inter_ipu_copies_buffer_size);
  auto resources = HloPoplarTestBase::GetMockResources(
      device, module, /*replication factor*/ 1, info);
  resources->merge_infeed_io_copies = merge_infeeds;

  auto target = resources->main_graph->getTarget();
  auto tiles_per_ipu = target.getTilesPerIPU();

  // Add mock vgraphs
  for (int i = 0; i < number_of_vgraphs; ++i) {
    resources->shard_compute_graphs.emplace_back(
        resources->main_graph->createVirtualGraph(i * tiles_per_ipu,
                                                  (i + 1) * tiles_per_ipu));
  }
  resources->shard_to_ipu_id.resize(number_of_vgraphs);
  absl::c_iota(resources->shard_to_ipu_id, 0);

  return std::move(resources);
}

std::unique_ptr<CompilerResources> HloPoplarTestBase::GetMockResources(
    poplar::Device& device, HloModule* module, int32 replication_factor,
    const CompilerInformation& info) {
  auto resources = CompilerResources::CreateTestDefault(module, info);
  resources->module_call_graph = CallGraph::Build(module);
  resources->CreateMainGraphAndPreamble(device.getTarget(), replication_factor);
  resources->replication_factor = replication_factor;
  resources->partition_replication_factor = replication_factor;

  poplin::addCodelets(*resources->main_graph);
  popnn::addCodelets(*resources->main_graph);
  popops::addCodelets(*resources->main_graph);
  poprand::addCodelets(*resources->main_graph);

  return resources;
}

StatusOr<int32> HloPoplarTestBase::GetMaxIpuCount() {
  const char* tf_ipu_count = getenv("TF_IPU_COUNT");
  if (!tf_ipu_count) {
    // Running without hardware, no IPUs available.
    return 0;
  }
  int32 count = std::atoi(tf_ipu_count);
  if (count < 0) {
    return InternalError("Invalid TF_IPU_COUNT value");
  }
  return count;
}

StatusOr<poplar::Engine> HloPoplarTestBase::Compile(
    CompilerResources& resources, HloModule* module) {
  VLOG(3) << "Compiling...";
  XLA_VLOG_LINES(1, module->ToString());

  EXPECT_TRUE(HloTrivialScheduler().Run(module).ValueOrDie());

  auto entry = module->entry_computation();
  auto order = module->schedule().sequence(entry).instructions();
  EntryVisitor visitor(resources, entry);

  TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, order));

  DriverProgramSequence main_program(*resources.main_graph);
  main_program.add(*resources.preamble_sequence);
  main_program.add(
      visitor.GetSequenceAndInitializeCounters(*resources.main_graph));

  return poplar::Engine(*resources.main_graph, main_program);
}

}  // namespace poplarplugin
}  // namespace xla

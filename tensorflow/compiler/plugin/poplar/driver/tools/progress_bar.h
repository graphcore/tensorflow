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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PROGRESS_BAR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PROGRESS_BAR_H_

#include <map>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

// Base class for progress bar.
class ProgressBarBase {
 public:
  virtual ~ProgressBarBase() = default;
  virtual void Start() = 0;
  virtual void MoveToNextStage() = 0;
  virtual void Update(const HloInstruction* inst) = 0;
  virtual void Update(std::size_t progress, std::size_t total) = 0;
  virtual void Finish() = 0;
};

// Mock implementation of the progress bar which doesn't do anything - can be
// used by default where/when the progress bar is not needed.
class NoProgressBar : public ProgressBarBase {
 public:
  void Start() override{};
  void MoveToNextStage() override{};
  void Update(const HloInstruction* inst) override{};
  void Update(std::size_t progress, std::size_t total) override{};
  void Finish() override{};
};

// Actual progress bar class.
class ProgressBar : public ProgressBarBase {
 public:
  explicit ProgressBar(const HloModule* module);
  ~ProgressBar() override;

  enum class CompilationStage {
    kHloOptimizations = 0,
    kPrePlanning,
    kGraphConstruction,
    kGraphCompilation,
    kFinished,
  };

  // Starts the progress bar - prints the setup to the console.
  void Start() override;

  // Unconditionally moves the progress to the current stage end point and
  // changes the current stage to the next stage.
  void MoveToNextStage() override;

  // Updates `current_stage_steps` by how many instructions were processed
  // (without this, during outlining/fusions we could miss some instructions).
  void Update(const HloInstruction* inst) override;

  // Updates the state.
  void Update(std::size_t progress, std::size_t total) override;

  // Marks compilation as finished.
  void Finish() override;

 private:
  const HloModule* module_;
  const std::size_t num_instructions_;

  std::mutex hlo_mu_;
  std::mutex progress_mu_;

  // Current progress out of 100.
  std::size_t progress_ GUARDED_BY(progress_mu_) = 0;
  // Starting progress postion for the current stage.
  std::size_t stage_start_progress_units_ GUARDED_BY(progress_mu_) = 0;
  // The current stage.
  CompilationStage current_stage_ GUARDED_BY(progress_mu_) =
      CompilationStage::kHloOptimizations;

  // HloInstructions which have been visited.
  absl::flat_hash_set<const HloInstruction*> visited_instructions_
      GUARDED_BY(hlo_mu_);

  // Describes how long each step is - derrived from experience - adds up to
  // 100.
  const std::map<CompilationStage, std::size_t> per_stage_time_units_ = {
      {CompilationStage::kHloOptimizations, 2},
      {CompilationStage::kPrePlanning, 2},
      {CompilationStage::kGraphConstruction, 36},
      {CompilationStage::kGraphCompilation, 60}};

  // Bar specific options.
  const std::size_t bar_width_ = 50;
  const char start_symbol_ = '[';
  const char end_symbol_ = ']';
  const char progress_symbol_ = '#';
  const char filler_symbol_ = ' ';
  const std::size_t bar_update_us_ = 100000;

  // Warning: The `finished_` member must outlive the `thread_` member.
  absl::Notification finished_;
  std::unique_ptr<tensorflow::Thread> thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(ProgressBar);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PROGRESS_BAR_H_

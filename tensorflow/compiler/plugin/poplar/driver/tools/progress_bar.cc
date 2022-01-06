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

#include "tensorflow/compiler/plugin/poplar/driver/tools/progress_bar.h"

#include <queue>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {
std::string CompilationStageToString(ProgressBar::CompilationStage stage) {
  switch (stage) {
    case ProgressBar::CompilationStage::kHloOptimizations: {
      return "Step: Hlo Optimizations";
    }
    case ProgressBar::CompilationStage::kPrePlanning: {
      return "Step: Pre-Planning Operations";
    }
    case ProgressBar::CompilationStage::kGraphConstruction: {
      return "Step: Graph Construction";
    }
    case ProgressBar::CompilationStage::kGraphCompilation: {
      return "Step: Graph Compilation";
    }
    case ProgressBar::CompilationStage::kFinished: {
      return "Compilation Finished";
    }
    default: {
      LOG(FATAL) << "Unknown compilation stage.";
      return "";
    }
  }
}

std::string ReadableTime(std::size_t time_us) {
  const std::size_t time_ms = time_us / 1000;
  const std::size_t miliseconds = time_ms % 1000;
  const std::size_t deciseconds = miliseconds / 100;

  const std::size_t time_s = time_ms / 1000;
  const std::size_t seconds = time_s % 60;

  const std::size_t time_m = time_s / 60;
  const std::size_t minutes = time_m % 60;

  const std::size_t time_h = time_m / 60;

  return absl::StrFormat("%02llu:%02llu:%02llu.%llu", time_h, minutes, seconds,
                         deciseconds);
}
};  // namespace

ProgressBar::ProgressBar(const HloModule* module)
    : module_(module), num_instructions_(module_->instruction_count()) {}

ProgressBar::~ProgressBar() {
  finished_.Notify();
  // The destructor of `thread_` will now block until the thread has joined.
}

// Starts the progress bar - prints the setup to the console.
void ProgressBar::Start() {
  std::cout << "Compiling module " << module_->name() << ":" << std::endl;
  tensorflow::Env* env = tensorflow::Env::Default();
  // Create a thread which updates the progress bar at some interval.
  thread_.reset(env->StartThread(
      tensorflow::ThreadOptions{}, "progress_bar_thread", [env, this]() {
        const std::size_t time_start_us = env->NowMicros();
        while (true) {
          {
            std::lock_guard<std::mutex> g(progress_mu_);
            const std::size_t time_now_us = env->NowMicros();
            const std::size_t elapsed_time_us = time_now_us - time_start_us;

            const std::size_t current_bar_position = static_cast<std::size_t>(
                std::floor(static_cast<float>(progress_) *
                           (static_cast<float>(bar_width_) / 100.f)));

            // Clear the current line and return to the begining.
            std::cout << "\33[2K" << start_symbol_;

            for (std::size_t i = 0; i != current_bar_position; ++i) {
              std::cout << progress_symbol_;
            }

            for (std::size_t i = current_bar_position; i != bar_width_; ++i) {
              std::cout << filler_symbol_;
            }

            std::cout << end_symbol_ << " " << progress_ << "% "
                      << CompilationStageToString(current_stage_)
                      << " [Elapsed: " << ReadableTime(elapsed_time_us) << "]";
            if (finished_.HasBeenNotified()) {
              std::cout << std::endl;
              break;
            } else {
              std::cout << "\r" << std::flush;
            }
          }

          env->SleepForMicroseconds(bar_update_us_);
        }
      }));
}

void ProgressBar::MoveToNextStage() {
  std::lock_guard<std::mutex> g(progress_mu_);
  progress_ =
      stage_start_progress_units_ + per_stage_time_units_.at(current_stage_);
  stage_start_progress_units_ = progress_;
  current_stage_ = static_cast<ProgressBar::CompilationStage>(
      static_cast<int>(current_stage_) + 1);
}

void ProgressBar::Update(const HloInstruction* inst) {
  std::lock_guard<std::mutex> g(hlo_mu_);
  // Find the number of instructions which this instruction should have
  // evaluated (they don't appear in the visited set). This occurs
  // due to outlining/fusions when we don't need to visit all the instructions.
  visited_instructions_.insert(inst);

  // Visit any called computations and their instructions.
  std::queue<const HloComputation*> comps;
  for (const HloComputation* comp : inst->called_computations()) {
    comps.push(comp);
  }

  while (!comps.empty()) {
    const HloComputation* comp = comps.front();
    comps.pop();
    for (const HloInstruction* to_visit : comp->instructions()) {
      if (visited_instructions_.contains(to_visit)) {
        continue;
      }
      visited_instructions_.insert(to_visit);
      for (const HloComputation* comp : to_visit->called_computations()) {
        comps.push(comp);
      }
    }
  }

  Update(visited_instructions_.size(), num_instructions_);
}

void ProgressBar::Update(std::size_t progress, std::size_t total) {
  std::lock_guard<std::mutex> g(progress_mu_);
  progress_ = stage_start_progress_units_ +
              static_cast<std::size_t>(std::floor(
                  static_cast<float>(per_stage_time_units_.at(current_stage_)) *
                  static_cast<float>(progress) / static_cast<float>(total)));
}

void ProgressBar::Finish() {
  std::lock_guard<std::mutex> g(progress_mu_);
  progress_ = 100;
  current_stage_ = CompilationStage::kFinished;
  // Using a notification here guarantees the memory order.
  finished_.Notify();
}

}  // namespace poplarplugin
}  // namespace xla

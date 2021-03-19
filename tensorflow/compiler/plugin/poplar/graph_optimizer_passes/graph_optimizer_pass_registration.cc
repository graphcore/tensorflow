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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/extract_outside_compilation_pass.h"
#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/gradient_accumulation_optimization_pass.h"
#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/reorder_gradient_accumulation_pass.h"
#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/static_shape_inference_pass.h"
#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/verify_gradient_accumulation_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// See tensorflow/compiler/jit/jit_compilation_pass_registration.cc for the
// other passes.

// Run this before ReorderGradientAccumulationPass (18).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 17,
                      ReorderGradientAccumulationPass);

// Run this after ReorderGradientAccumulationPass (17) and before
// GradientAccumulationOptimizationPass(20).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 18,
                      VerifyGradientAccumulationPass);

// Run this before StaticShapeInferencePass (20).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 19,
                      GradientAccumulationOptimizationPass);

// This must run before EncapsulateXlaComputationsPass (26).
// We need static shapes for the outside compilation scope in order to know what
// shapes we must receive on the IPU. Needs to run before XLA encapsulation
// since it does not handle function calls.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 20,
                      StaticShapeInferencePass);

// Run this between EncapsulateSubgraphsPass (50) and BuildXlaOpsPass (60).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 55,
                      ExtractOutsideCompilationPass);

}  // namespace tensorflow

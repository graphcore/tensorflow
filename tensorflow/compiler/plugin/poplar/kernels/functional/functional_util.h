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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_FUNCTIONAL_FUNCTIONAL_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_FUNCTIONAL_FUNCTIONAL_UTIL_H_
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace poplarplugin {
// Return the default compilation comptions for functions.
XlaCompiler::CompileOptions GetDefaultCompileOptions();

// Function which tries to get all the arguments to the Op. It optionally tries
// to evaluate any constant inputs to a value so that they can be propagated.
xla::StatusOr<std::vector<XlaCompiler::Argument>> GetXlaArguments(
    XlaOpKernelContext* ctx, const DataTypeVector& input_types,
    bool evaluate_constants = true);

xla::StatusOr<XlaCompiler::Argument> GetXlaArgument(
    XlaOpKernelContext* ctx, size_t input, bool evaluate_constants = true);

// Return the number of kResource type XlaArguments in the given vector.
int CountResourceArgs(const std::vector<XlaCompiler::Argument>& arguments);

// Function which gets all non-constant function inputs.
xla::StatusOr<std::vector<xla::XlaOp>> GetXlaInputs(
    XlaOpKernelContext* ctx,
    const std::vector<XlaCompiler::Argument>& arguments,
    const std::vector<int>& input_mapping);

// Same as XlaCompiler::CompileFunction, but will recompile if there are any
// TensorArray gradients which have been accessed.
Status CompileFunction(XlaOpKernelContext* ctx,
                       const XlaCompiler::CompileOptions& options,
                       const NameAttrList& fn_name_attrs,
                       std::vector<XlaCompiler::Argument>& args,
                       XlaCompiler::CompilationResult* result);

// Base kernal class for implementing functional ops which do not output
// resources unless modified from the called computations.
class FunctionBaseOp : public XlaOpKernel {
 public:
  explicit FunctionBaseOp(OpKernelConstruction* ctx);
  FunctionBaseOp(OpKernelConstruction* ctx, bool evaluate_constants);

  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  virtual Status SetConfig(xla::XlaBuilder* builder, xla::XlaOp& operation) = 0;
  virtual xla::StatusOr<std::vector<XlaCompiler::Argument>> GetArguments(
      XlaOpKernelContext* ctx) const;

 private:
  const bool evaluate_constants_;
  const NameAttrList* to_apply_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
};
}  // namespace poplarplugin
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_FUNCTIONAL_FUNCTIONAL_UTIL_H_

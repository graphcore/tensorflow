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
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/kernels/application_runtime/resource_handle_pruner.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

Status BuildCompilationCache(OpKernelContext* ctx, se::Platform* platform,
                             XlaCompilationCache** out_cache) {
  xla::LocalClientOptions client_options;
  client_options.set_platform(platform);
  client_options.set_intra_op_parallelism_threads(
      ctx->device()->tensorflow_cpu_worker_threads()->num_threads);
  TF_ASSIGN_OR_RETURN(
      auto* client, xla::ClientLibrary::GetOrCreateLocalClient(client_options));
  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice("IPU", &registration)) {
    return errors::InvalidArgument("No JIT device registered for IPU");
  }

  *out_cache = new XlaCompilationCache(
      client, DeviceType(registration->compilation_device_name));
  return Status::OK();
}

xla::StatusOr<xla::LocalExecutable*> CompileExecutable(
    OpKernelContext* ctx, const NameAttrList& function, se::Platform* platform,
    absl::Span<const Tensor* const> inputs,
    absl::Span<const VariableInfo> variable_infos,
    absl::Span<const int> constants) {
  auto* resource_manager = ctx->resource_manager();
  if (!resource_manager) {
    return errors::Internal("Resource manager not found");
  }

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(resource_manager->LookupOrCreate<XlaCompilationCache>(
      resource_manager->default_container(), "ipu_application_compile_cache",
      &cache, [&](XlaCompilationCache** cache) {
        return BuildCompilationCache(ctx, platform, cache);
      }));
  core::ScopedUnref cache_ref(cache);

  const auto* function_library = ctx->function_library();
  if (!function_library) {
    return errors::Internal("Function library not found");
  }

  const auto* flib_def = function_library->GetFunctionLibraryDefinition();
  const auto* func_def = CHECK_NOTNULL(flib_def)->Find(function.name());
  if (!func_def) {
    return errors::Internal("Function not found: " + function.name());
  }

  VLOG(1) << "Compiling function: " << DebugString(*func_def);

  XlaCompiler::Options options;
  options.client = cache->client();
  options.device_type = cache->device_type();
  options.flib_def = flib_def;
  options.graph_def_version = function_library->graph_def_version();

  options.device_allocator = std::make_shared<se::TfAllocatorAdapter>(
      ctx->device()->GetAllocator({}), platform);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  compile_options.always_return_tuple = false;

  // IPU Specific - store the names of all inputs.
  std::vector<std::string> mangled_input_names(inputs.size());
  for (int64 i = 0; i != inputs.size(); ++i) {
    mangled_input_names[i] = ctx->op_kernel().requested_input(i);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<XlaCompiler::Argument> arguments,
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constants, inputs, variable_infos,
          static_cast<Device*>(ctx->device()), mangled_input_names));

  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;
  TF_RETURN_IF_ERROR(cache->Compile(options, function, arguments,
                                    compile_options,
                                    XlaCompilationCache::CompileMode::kStrict,
                                    &compilation_result, &executable));
  return executable;
}

std::vector<const Tensor*> FilterResourceTensors(
    const std::vector<const Tensor*> src) {
  std::vector<const Tensor*> result;
  std::copy_if(
      src.cbegin(), src.cend(), std::back_inserter(result),
      [](const Tensor* tensor) { return tensor->dtype() != DT_RESOURCE; });

  return result;
}

}  // namespace

class IPUApplicationCompile : public OpKernel {
 public:
  explicit IPUApplicationCompile(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &function_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_indices", &resource_indices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("constant_indices", &constant_indices_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("executable_output_path", &executable_output_path_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("prune_resource_tensors", &prune_resource_tensors_));
  }

  void Compute(OpKernelContext* ctx) {
    auto platform_or_status =
        se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES_OK(ctx, platform_or_status.status());
    auto* platform = platform_or_status.ValueOrDie();

    std::vector<const Tensor*> inputs = InputsFromContext(ctx);
    std::vector<VariableInfo> variable_infos;

    // We indiscriminately recursively destroy all resource tensors.
    // Option should be used when resources are frozen. Given that assumption,
    // any remaining resources still inside the function shouldn't be there.
    // FuncGraph creates a resource in the function when it captures an eager
    // resource. It can later freeze it by pulling its ReadVariableOp outside
    // the function, but at that point the captured resource has already been
    // created. Resource placeholders left by FuncGraph should be removed.
    if (prune_resource_tensors_) {
      CHECK(resource_indices_.empty());
      inputs = FilterResourceTensors(inputs);
      constant_indices_ = std::vector<int>(inputs.size());
      // Fix constant indicies, since resource inputs are skipped
      // it will have some holes e.g. [0,1,3], indicies should be continous
      std::iota(constant_indices_.begin(), constant_indices_.end(), 0);
      ResourceHandlePruner pruner{ctx->function_library()};
      OP_REQUIRES_OK(ctx, pruner.Run(function_));
    }

    OP_REQUIRES_OK(ctx, GetVariableInfosFromInputs(
                            ctx->resource_manager(), ctx->device(), inputs,
                            resource_indices_, &variable_infos));

    OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variable_infos)));

    auto executable_or_status = CompileExecutable(
        ctx, function_, platform, inputs, variable_infos, constant_indices_);
    OP_REQUIRES_OK(ctx, executable_or_status.status());

    auto* poplar_executable =
        dynamic_cast<xla::poplarplugin::PoplarExecutable*>(
            executable_or_status.ValueOrDie()->executable());
    OP_REQUIRES(ctx, poplar_executable != nullptr,
                errors::Internal("Missing Poplar executable"));

    OP_REQUIRES_OK(ctx, poplar_executable->Serialize(executable_output_path_));
    ctx->set_output(0, Tensor(executable_output_path_));
  }

 private:
  NameAttrList function_;
  std::string executable_output_path_;
  std::vector<int> constant_indices_;
  std::vector<int> resource_indices_;
  bool prune_resource_tensors_;

  TF_DISALLOW_COPY_AND_ASSIGN(IPUApplicationCompile);
};

REGISTER_KERNEL_BUILDER(Name("IPUApplicationCompile").Device(DEVICE_CPU),
                        IPUApplicationCompile);

}  // namespace tensorflow

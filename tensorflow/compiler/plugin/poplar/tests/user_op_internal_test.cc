/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include <iostream>
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

// No longer experimental as of C++17
#include <experimental/filesystem>

namespace xla {
namespace poplarplugin {
namespace {

using UserOperator = HloTestBase;

// Check default values
TEST_F(UserOperator, UserOpDefaults) {
  void* function_ptr = nullptr;
  void* metadata_ptr = nullptr;
  void* allocator_ptr = nullptr;

  xla::Shape shape;

  // Call into the instruction directly.
  const int64 gradient_size = 0;
  const int64 partial_derivative_index = 0;
  const bool is_user_read_write = false;
  const std::string attributes;
  HloUserOpInstruction inst{{},
                            shape,
                            "my_gc_file.gc",
                            function_ptr,
                            metadata_ptr,
                            allocator_ptr,
                            gradient_size,
                            partial_derivative_index,
                            is_user_read_write,
                            attributes};

  EXPECT_EQ(inst.NumInputs(), 0);
  EXPECT_EQ(inst.IsPopOpsElementwise(), false);
  EXPECT_EQ(inst.IsGradient(), false);
  EXPECT_EQ(inst.GetPointerToFunc(), nullptr);
  EXPECT_EQ(inst.GetAllocatorFunc(), nullptr);
  EXPECT_EQ(inst.GetPath(), "my_gc_file.gc");

  absl::flat_hash_set<int64> alloc_indices = inst.AllocatingIndices();
  absl::flat_hash_set<int64> replica_identical_outputs =
      inst.ReplicaIdenticalOutputIndices();
  absl::flat_hash_map<int64, int64> layout_deps = inst.LayoutDependencies();

  EXPECT_TRUE(alloc_indices.empty());
  EXPECT_TRUE(replica_identical_outputs.empty());
  EXPECT_TRUE(layout_deps.empty());
}

// Check default values
TEST_F(UserOperator, UserOpReadMetadata) {
  xla::Shape shape;

  // Path to our test.so
  std::string lib_path =
      std::experimental::filesystem::current_path().string() +
      "/tensorflow/compiler/plugin/poplar/libuser_op_test.so";

  // Load the shared library.
  void* handle;
  tensorflow::Env::Default()->LoadLibrary(lib_path.c_str(), &handle);

  void* function_ptr = nullptr;

  // Get the symbol from the library for each of the functions.
  tensorflow::Env::Default()->GetSymbolFromLibrary(handle, "Build",
                                                   &function_ptr);
  ASSERT_TRUE(function_ptr);

  void* metadata_ptr = nullptr;
  tensorflow::Env::Default()->GetSymbolFromLibrary(handle, "Build_metadata",
                                                   &metadata_ptr);
  ASSERT_TRUE(metadata_ptr);

  void* allocator_ptr = nullptr;
  tensorflow::Env::Default()->GetSymbolFromLibrary(handle, "Build_allocator",
                                                   &allocator_ptr);
  ASSERT_TRUE(allocator_ptr);

  // Call into the instruction directly.
  const int64 gradient_size = 0;
  const int64 partial_derivative_index = 0;
  const bool is_user_read_write = false;
  const std::string attributes;
  HloUserOpInstruction inst{{},
                            shape,
                            "my_gc_file.gc",
                            function_ptr,
                            metadata_ptr,
                            allocator_ptr,
                            gradient_size,
                            partial_derivative_index,
                            is_user_read_write,
                            attributes};

  EXPECT_EQ(inst.NumInputs(), 0);
  EXPECT_EQ(inst.IsPopOpsElementwise(), true);
  EXPECT_EQ(inst.IsGradient(), false);
  EXPECT_NE(inst.GetPointerToFunc(), nullptr);
  EXPECT_NE(inst.GetAllocatorFunc(), nullptr);
  EXPECT_EQ(inst.GetPath(), "my_gc_file.gc");

  absl::flat_hash_set<int64> alloc_indices = inst.AllocatingIndices();

  EXPECT_EQ(alloc_indices.size(), 4);
  EXPECT_TRUE(alloc_indices.contains(0));
  EXPECT_TRUE(alloc_indices.contains(1));
  EXPECT_TRUE(alloc_indices.contains(2));
  EXPECT_TRUE(alloc_indices.contains(3));

  absl::flat_hash_set<int64> replica_identical_outputs =
      inst.ReplicaIdenticalOutputIndices();
  EXPECT_EQ(replica_identical_outputs.size(), 3);
  EXPECT_TRUE(replica_identical_outputs.contains(0));
  EXPECT_TRUE(replica_identical_outputs.contains(1));
  EXPECT_TRUE(replica_identical_outputs.contains(2));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

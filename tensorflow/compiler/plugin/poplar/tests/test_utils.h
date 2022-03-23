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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_TEST_UTILS_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_TEST_UTILS_H

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
#ifdef XLA_TEST_BACKEND_POPLAR
#define POPLAR_TEST_P(X, Y) TEST_P(X, Y)
#else
#define POPLAR_TEST_P(X, Y)
#endif

#define MAKE_HLO_TEST_CASE(hlo_string) HloTestCase(#hlo_string, hlo_string)

// Common types/utilities for writing HLO based tests
struct HloTestFixture : HloTestBase {
  using HloTestBase::HloTestBase;

  ::testing::AssertionResult SetUpHloModule(const std::string& hlo,
                                            int64 replica_count = 1) {
    auto config = GetModuleConfigForTest();
    config.set_replica_count(replica_count);
    auto module = ParseAndReturnVerifiedModule(hlo, config);
    if (module.ok()) {
      hlo_module_owner_ = std::move(module.ValueOrDie());
      hlo_module_ = hlo_module_owner_.get();

      annotations_ = absl::make_unique<CompilerAnnotations>(hlo_module_);

      return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "Parsing hlo failed: " << module.status().error_message();
  }

  HloInstruction* FindRootInstruction() {
    auto* entry_comp = hlo_module_->entry_computation();
    return entry_comp->root_instruction();
  }

  VerifiedHloModule* hlo_module_ = nullptr;
  std::unique_ptr<CompilerAnnotations> annotations_;

  std::unique_ptr<VerifiedHloModule> hlo_module_owner_;
};

struct HloTestCase {
  HloTestCase(const std::string& name, const std::string& hlo)
      : name(name), hlo(hlo), replica_count(1) {}
  HloTestCase(const std::string& name, const std::string& hlo,
              int64 replica_count)
      : name(name), hlo(hlo), replica_count(replica_count) {}
  std::string name;
  std::string hlo;
  int64 replica_count;
};

std::ostream& operator<<(std::ostream& stream, const HloTestCase& test_case) {
  stream << test_case.name;
  return stream;
}

template <class Base = HloTestFixture>
struct ParameterizedHloTestFixture
    : Base,
      ::testing::WithParamInterface<HloTestCase> {
  void SetUp() override {
    ASSERT_TRUE(Base::SetUpHloModule(GetParam().hlo, GetParam().replica_count));
  }
};

// Utility for setting the name of parameterized tests from the
// HloTestCase.
std::string HloTestCaseName(const ::testing::TestParamInfo<HloTestCase>& info) {
  return info.param.name;
}

bool HasOperand(const HloInstruction* parent, const HloInstruction* arg) {
  for (const auto* inst : parent->operands()) {
    if (inst == arg) return true;
  }
  return false;
}

bool HasOperandIn(const HloInstruction* parent,
                  const std::set<const HloInstruction*> arg_list) {
  for (const auto* inst : parent->operands()) {
    if (arg_list.count(inst) == 1) return true;
  }
  return false;
}

struct TemporaryDirManager {
 public:
  explicit TemporaryDirManager(const std::string& dir_name)
      : dir_name_(dir_name) {
    if (!tensorflow::Env::Default()->LocalTempFilename(&dir_name_)) {
      LOG(FATAL) << "Could not create a temporary directory.";
    }
    TF_CHECK_OK(tensorflow::Env::Default()->CreateDir(dir_name_));
  }
  ~TemporaryDirManager() {
    tensorflow::int64 undeleted_dirs, undeleted_files;
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteRecursively(
        dir_name_, &undeleted_dirs, &undeleted_files));
  }
  const std::string& GetDirName() { return dir_name_; }

 private:
  std::string dir_name_;
};

template <typename Instruction>
int64 GetNumInstructions(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(), [](const HloInstruction* inst) {
    return DynCast<Instruction>(inst);
  });
}

namespace reference_util {
// Implementations of 3D functions which are missing from the reference util.
std::vector<float> Reduce3DTo1D(
    const Array3D<float>& array, float init, absl::Span<const int64> dims,
    const std::function<float(float, float)>& reduce_function) {
  std::vector<float> result;
  CHECK_EQ(dims.size(), 2);
  const std::set<int64> dim_set(dims.begin(), dims.end());
  CHECK_EQ(dim_set.size(), 2);
  for (int64 a0 = 0; a0 == 0 || (!dim_set.count(0) && a0 < array.n1()); ++a0) {
    for (int64 a1 = 0; a1 == 0 || (!dim_set.count(1) && a1 < array.n2());
         ++a1) {
      for (int64 a2 = 0; a2 == 0 || (!dim_set.count(2) && a2 < array.n3());
           ++a2) {
        float accumulator = init;
        for (int64 i0 = 0; i0 == 0 || (dim_set.count(0) && i0 < array.n1());
             ++i0) {
          for (int64 i1 = 0; i1 == 0 || (dim_set.count(1) && i1 < array.n2());
               ++i1) {
            for (int64 i2 = 0; i2 == 0 || (dim_set.count(2) && i2 < array.n3());
                 ++i2) {
              // Handle zero-sized arrays.
              if (array.n1() > 0 && array.n2() > 0 && array.n3() > 0) {
                accumulator = reduce_function(accumulator,
                                              array(a0 + i0, a1 + i1, a2 + i2));
              }
            }
          }
        }
        result.push_back(accumulator);
      }
    }
  }
  return result;
}

std::unique_ptr<Array3D<float>> Broadcast1DTo3D(
    const std::vector<float>& array, const std::vector<int64>& bounds,
    int64 broadcast_from_dim) {
  auto result =
      absl::make_unique<Array3D<float>>(bounds[0], bounds[1], bounds[2]);
  for (int64 i = 0; i < result->n1(); ++i) {
    for (int64 j = 0; j < result->n2(); ++j) {
      for (int64 k = 0; k < result->n3(); ++k) {
        switch (broadcast_from_dim) {
          case 0:
            (*result)(i, j, k) = array[i];
            break;
          case 1:
            (*result)(i, j, k) = array[j];
            break;
          case 2:
            (*result)(i, j, k) = array[k];
            break;
          default:
            break;
        }
      }
    }
  }
  return result;
}

// Applies map_function to each element in the input (3D array) and returns
// the result.
// (n1, n2, n3) index of each element is also provided as
// arguments to map_function.
template <typename F>
static std::unique_ptr<Array3D<float>> MapWithIndexArray3D(
    const Array3D<float>& input, F&& map_function) {
  auto result =
      absl::make_unique<Array3D<float>>(input.n1(), input.n2(), input.n3());
  for (int64 n1 = 0; n1 < input.n1(); ++n1) {
    for (int64 n2 = 0; n2 < input.n2(); ++n2) {
      for (int64 n3 = 0; n3 < input.n3(); ++n3) {
        (*result)(n1, n2, n3) = map_function(input(n1, n2, n3), n1, n2, n3);
      }
    }
  }
  return result;
}

// Applies map_function to each element in the input (3D array) and returns
// the result.
template <typename F>
static std::unique_ptr<Array3D<float>> MapArray3D(const Array3D<float>& input,
                                                  F&& map_function) {
  return MapWithIndexArray3D(input, [&](float value, int64, int64, int64) {
    return map_function(value);
  });
}

// Applies map_function to each pair of element in lhs and rhs (3D array) and
// returns the result.
// (n1, n2, n3) index of each element is also provided as
// arguments to map_function.
template <typename F>
static std::unique_ptr<Array3D<float>> MapWithIndexArray3D(
    const Array3D<float>& lhs, const Array3D<float>& rhs, F&& map_function) {
  auto result = absl::make_unique<Array3D<float>>(lhs.n1(), lhs.n2(), lhs.n3());
  for (int64 n1 = 0; n1 < lhs.n1(); ++n1) {
    for (int64 n2 = 0; n2 < lhs.n2(); ++n2) {
      for (int64 n3 = 0; n3 < lhs.n3(); ++n3) {
        (*result)(n1, n2, n3) =
            map_function(lhs(n1, n2, n3), rhs(n1, n2, n3), n1, n2, n3);
      }
    }
  }
  return result;
}

// Applies map_function to each pair of elements in the input lhs and rhs
// (3D array) and returns the result.
template <typename F>
static std::unique_ptr<Array3D<float>> MapArray3D(const Array3D<float>& lhs,
                                                  const Array3D<float>& rhs,
                                                  F&& map_function) {
  return MapWithIndexArray3D(lhs, rhs,
                             [&](float lhs, float rhs, int64, int64, int64) {
                               return map_function(lhs, rhs);
                             });
}

static std::unique_ptr<Array3D<float>> BatchNorm3D(const Array3D<float>& input,
                                                   const Array3D<float>& mean,
                                                   const Array3D<float>& var,
                                                   const Array3D<float>& scale,
                                                   const Array3D<float>& offset,
                                                   float epsilon) {
  auto normalized = *reference_util::MapArray3D(
      input, mean, [](float a, float b) { return a - b; });
  normalized = *reference_util::MapArray3D(
      normalized, var,
      [&](float a, float b) { return a / std::sqrt(b + epsilon); });
  normalized = *reference_util::MapArray3D(
      normalized, scale, [](float a, float b) { return a * b; });
  return reference_util::MapArray3D(normalized, offset,
                                    [](float a, float b) { return a + b; });
}

}  // namespace reference_util
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TEST_TEST_UTILS_H

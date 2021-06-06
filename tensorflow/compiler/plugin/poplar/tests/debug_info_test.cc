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
===============================================================*/

#include <fstream>

#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"

struct TemporaryFileManager {
 public:
  explicit TemporaryFileManager(const std::string& file_name)
      : file_name_(file_name) {
    if (!tensorflow::Env::Default()->LocalTempFilename(&file_name_)) {
      LOG(FATAL) << "Could not create a file.";
    }
  }

  ~TemporaryFileManager() {
    auto status = tensorflow::Env::Default()->DeleteFile(file_name_);
    if (!status.ok()) {
      LOG(FATAL) << status.ToString();
    }
  }
  const std::string& GetFileName() { return file_name_; }

 private:
  std::string file_name_;
};

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<Json::Value> ReadJsonFile(const std::string& file_name) {
  std::ifstream file(file_name);
  Json::Value output;
  Json::Reader reader;
  if (!reader.parse(file, output)) {
    return InternalError("Could not parse json file.");
  }
  return output;
}

using DebugInfoTest = HloTestBase;

TEST_F(DebugInfoTest, TestXlaOpDebugInfo) {
  TemporaryFileManager tfm("file.json");
  poplar::DebugInfo::initializeStreamer(tfm.GetFileName(),
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debug_context(
        "test", poplar::SourceLocation("function_name", "file_name", 42));
    xla::OpMetadata metadata;
    metadata.set_op_type("fred");
    metadata.set_op_name("test");
    XlaOpDebugInfo debug_info(debug_context, metadata);
  }

  poplar::DebugInfo::closeStreamer();

  TF_ASSERT_OK_AND_ASSIGN(auto root, ReadJsonFile(tfm.GetFileName()));

  EXPECT_EQ(1, root["contexts"].size());
  auto c = root["contexts"][0];
  EXPECT_EQ("xla_op", c["layer"].asString());
  EXPECT_EQ("op", c["category"].asString());
  EXPECT_EQ("test", c["op_name"].asString());
  EXPECT_EQ("fred", c["op_type"].asString());
  EXPECT_FALSE(c.isMember("sourcefile"));
  EXPECT_FALSE(c.isMember("sourceline"));
}

TEST_F(DebugInfoTest, TestHloInstructionDebugInfo_Constant) {
  TemporaryFileManager tfm("file.json");
  poplar::DebugInfo::initializeStreamer(tfm.GetFileName(),
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debug_context(
        poplar::SourceLocation("function_name", "file_name", 42));
    auto instruction = HloInstruction::CreateConstant(xla::Literal());
    HloInstructionDebugInfo debug_info(debug_context, instruction.get());
  }

  poplar::DebugInfo::closeStreamer();

  TF_ASSERT_OK_AND_ASSIGN(auto root, ReadJsonFile(tfm.GetFileName()));

  EXPECT_EQ(1, root["contexts"].size());
  auto c = root["contexts"][0];
  EXPECT_EQ("hloinstruction", c["layer"].asString());
  EXPECT_EQ("constant", c["hlo_name"].asString());
  EXPECT_EQ(-1, c["hlo_id"].asInt64());
  EXPECT_EQ("constant", c["opcode"].asString());
  EXPECT_EQ("() -> ()", c["signature"].asString());
  EXPECT_EQ("%constant = () constant({...})", c["debug_string"].asString());
  EXPECT_EQ(0, c["operand_count"].asInt());
  EXPECT_EQ(0, c["operands"].size());
  EXPECT_EQ(0, c["users"].size());
}

TEST_F(DebugInfoTest, TestHloInstructionDebugInfo_Dot) {
  TemporaryFileManager tfm("file.json");
  poplar::DebugInfo::initializeStreamer(tfm.GetFileName(),
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debug_context(
        poplar::SourceLocation("function_name", "file_name", 42));

    Shape r1f32 = ShapeUtil::MakeShape(F32, {1});
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_batch_dimensions(0);
    dot_dnums.add_rhs_batch_dimensions(0);

    auto x = HloInstruction::CreateParameter(0, r1f32, "x");
    auto y = HloInstruction::CreateParameter(1, r1f32, "y");

    // Need to create a root node for a computation so we can get computation
    // added to the instruction below. This is just so we can test for the
    // presence of the fields.
    auto root = HloInstruction::CreateDot(r1f32, x.get(), y.get(), dot_dnums,
                                          DefaultPrecisionConfig(2));
    auto builder = HloComputation::Builder("ComputationA");
    builder.AddInstruction(std::move(root));

    auto instruction = HloInstruction::CreateDot(
        r1f32, x.get(), y.get(), dot_dnums, DefaultPrecisionConfig(2));

    auto computation = builder.Build(root.get());
    auto a = computation->AddInstruction(std::move(instruction));
    HloInstructionDebugInfo debug_info(debug_context, a);
  }

  poplar::DebugInfo::closeStreamer();

  TF_ASSERT_OK_AND_ASSIGN(auto root, ReadJsonFile(tfm.GetFileName()));

  EXPECT_EQ(1, root["contexts"].size());
  auto c = root["contexts"][0];
  EXPECT_EQ("hloinstruction", c["layer"].asString());
  EXPECT_EQ("dot", c["hlo_name"].asString());
  EXPECT_EQ(-1, c["hlo_id"].asInt64());
  EXPECT_EQ("dot", c["opcode"].asString());
  EXPECT_EQ("(f32[1], f32[1]) -> f32[1]", c["signature"].asString());
  EXPECT_EQ(
      "%dot = f32[1]{0} dot(f32[1]{0} %x, f32[1]{0} %y), lhs_batch_dims={0}, "
      "lhs_contracting_dims={}, rhs_batch_dims={0}, rhs_contracting_dims={}",
      c["debug_string"].asString());
  EXPECT_EQ(2, c["operand_count"].asInt());
  EXPECT_EQ(2, c["operands"].size());
  EXPECT_EQ(0, c["users"].size());

  EXPECT_EQ("ComputationA", c["computation"].asString());
  EXPECT_EQ(-1, c["computation_id"].asInt64());
}

TEST_F(DebugInfoTest, TestPoplarOpDefDebugInfo) {
  TemporaryFileManager tfm("file.json");
  poplar::DebugInfo::initializeStreamer(tfm.GetFileName(),
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debug_context(
        poplar::SourceLocation("function_name", "file_name", 42));
    PoplarOpDefDebugInfo debug_info(debug_context, "SomeClass");
  }

  poplar::DebugInfo::closeStreamer();

  TF_ASSERT_OK_AND_ASSIGN(auto root, ReadJsonFile(tfm.GetFileName()));

  EXPECT_EQ(1, root["contexts"].size());
  auto c = root["contexts"][0];
  EXPECT_EQ("poplar_driver", c["layer"].asString());
  EXPECT_EQ("SomeClass", c["class"].asString());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

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

#include <stdio.h>
#include <iostream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

class TemporaryFileManager {
 public:
  explicit TemporaryFileManager(const char* ext) {
    // create temporary file name
    name = std::string(std::tmpnam(nullptr)) + "." + ext;
  }

  std::string name;

  ~TemporaryFileManager() {
    // delete the file
    remove(name.c_str());
  }
};

// Short alias for this namespace
namespace pt = boost::property_tree;

namespace xla {
namespace poplarplugin {
namespace {

using DebugInfoTest = HloTestBase;

TEST_F(DebugInfoTest, TestXlaOpDebugInfo) {
  TemporaryFileManager tfm("json");
  poplar::DebugInfo::initializeStreamer(tfm.name,
                                        poplar::DebugSerializationFormat::JSON);

  std::cout << tfm.name << std::endl;
  {
    poplar::DebugContext debugContext(
        "test", poplar::SourceLocation("function_name", "file_name", 42));
    xla::OpMetadata metadata;
    metadata.set_op_type("fred");
    metadata.set_op_name("test");
    XlaOpDebugInfo debugInfo(debugContext, metadata);
  }

  poplar::DebugInfo::closeStreamer();

  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(tfm.name, root);

  EXPECT_EQ(1, root.get_child("contexts").size());

  auto c = root.get_child("contexts").front().second;
  EXPECT_EQ("xla_op", c.get_child("layer").get_value<std::string>());
  EXPECT_EQ("op", c.get_child("category").get_value<std::string>());

  EXPECT_EQ("test", c.get_child("op_name").get_value<std::string>());
  EXPECT_EQ("fred", c.get_child("op_type").get_value<std::string>());

  EXPECT_FALSE(c.get_child_optional("sourcefile"));
  EXPECT_FALSE(c.get_child_optional("sourceline"));
}

TEST_F(DebugInfoTest, TestHloInstructionDebugInfo_Constant) {
  TemporaryFileManager tfm("json");
  poplar::DebugInfo::initializeStreamer(tfm.name,
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debugContext(
        poplar::SourceLocation("function_name", "file_name", 42));
    auto instruction = HloInstruction::CreateConstant(xla::Literal());
    HloInstructionDebugInfo debugInfo(debugContext, instruction.get());
  }

  poplar::DebugInfo::closeStreamer();

  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(tfm.name, root);

  EXPECT_EQ(1, root.get_child("contexts").size());
  auto c = root.get_child("contexts").front().second;
  EXPECT_EQ("hloinstruction", c.get_child("layer").get_value<std::string>());
  EXPECT_EQ("constant", c.get_child("hlo_name").get_value<std::string>());
  EXPECT_EQ(-1, c.get_child("hlo_id").get_value<int64>());
  EXPECT_EQ("constant", c.get_child("opcode").get_value<std::string>());
  EXPECT_EQ("() -> ()", c.get_child("signature").get_value<std::string>());
  EXPECT_EQ("%constant = () constant({...})",
            c.get_child("debug_string").get_value<std::string>());

  EXPECT_EQ(0, c.get_child("operand_count").get_value<int>());

  EXPECT_EQ(0, c.get_child("operands").size());
  EXPECT_EQ(0, c.get_child("users").size());
}

TEST_F(DebugInfoTest, TestHloInstructionDebugInfo_Dot) {
  TemporaryFileManager tfm("json");
  poplar::DebugInfo::initializeStreamer(tfm.name,
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debugContext(
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
    HloInstructionDebugInfo debugInfo(debugContext, a);
  }

  poplar::DebugInfo::closeStreamer();

  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(tfm.name, root);

  EXPECT_EQ(1, root.get_child("contexts").size());
  auto c = root.get_child("contexts").front().second;
  EXPECT_EQ("hloinstruction", c.get_child("layer").get_value<std::string>());
  EXPECT_EQ("dot", c.get_child("hlo_name").get_value<std::string>());
  EXPECT_EQ(-1, c.get_child("hlo_id").get_value<int64>());
  EXPECT_EQ("dot", c.get_child("opcode").get_value<std::string>());
  EXPECT_EQ("(f32[1], f32[1]) -> f32[1]",
            c.get_child("signature").get_value<std::string>());
  EXPECT_EQ(
      "%dot = f32[1]{0} dot(f32[1]{0} %x, f32[1]{0} %y), lhs_batch_dims={0}, "
      "lhs_contracting_dims={}, rhs_batch_dims={0}, rhs_contracting_dims={}",
      c.get_child("debug_string").get_value<std::string>());

  EXPECT_EQ(2, c.get_child("operand_count").get_value<int>());

  EXPECT_EQ("ComputationA",
            c.get_child("computation").get_value<std::string>());
  EXPECT_EQ(-1, c.get_child("computation_id").get_value<int64>());

  EXPECT_EQ(2, c.get_child("operands").size());
  EXPECT_EQ(0, c.get_child("users").size());
}

TEST_F(DebugInfoTest, TestPoplarOpDefDebugInfo) {
  TemporaryFileManager tfm("json");
  poplar::DebugInfo::initializeStreamer(tfm.name,
                                        poplar::DebugSerializationFormat::JSON);

  {
    poplar::DebugContext debugContext(
        poplar::SourceLocation("function_name", "file_name", 42));
    PoplarOpDefDebugInfo debugInfo(debugContext, "SomeClass");
  }

  poplar::DebugInfo::closeStreamer();

  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(tfm.name, root);

  EXPECT_EQ(1, root.get_child("contexts").size());

  auto c = root.get_child("contexts").front().second;
  EXPECT_EQ("poplar_driver", c.get_child("layer").get_value<std::string>());
  EXPECT_EQ("SomeClass", c.get_child("class").get_value<std::string>());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

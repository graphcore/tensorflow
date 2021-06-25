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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tensorflow/compiler/plugin/poplar/config/ipu_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

#include "google/protobuf/util/message_differencer.h"

namespace tensorflow {
namespace ipu {

// Utilities for using IpuConfigs with gmock Matchers.
bool operator==(const IpuConfig& lhs, const IpuConfig& rhs) {
  return google::protobuf::util::MessageDifferencer::Equals(lhs.options,
                                                            rhs.options);
}

std::ostream& operator<<(std::ostream& ostream, const IpuConfig& config) {
  ostream << config.options.DebugString();
  return ostream;
}

namespace {

#define ASSERT_STATUS_OK(status) \
  ASSERT_TRUE(status.ok()) << status.error_message();
#define ASSERT_STATUS_BAD(status) ASSERT_FALSE(status.ok())

Status RunSimpleGraph(const std::string& device_spec) {
  namespace tf = tensorflow;

  tf::Scope root = tf::Scope::NewRootScope().WithAssignedDevice(device_spec);
  tf::Output a =
      tf::ops::Placeholder(root.WithOpName("A"), tensorflow::DT_FLOAT);
  tf::Output a_squared = tf::ops::MatMul(root.WithOpName("Matmul"), a, a);
  if (!root.ok()) {
    return root.status();
  }

  tf::GraphDef graph;
  root.graph()->ToGraphDef(&graph);

  tf::Session* raw_session = nullptr;
  TF_RETURN_IF_ERROR(tf::NewSession(/*SessionOptions*/ {}, &raw_session));
  CHECK(raw_session);
  std::unique_ptr<tf::Session> session(raw_session);

  TF_RETURN_IF_ERROR(session->Create(graph));

  tf::Tensor input(tf::DT_FLOAT, tf::TensorShape({2, 2}));
  tf::test::FillValues<float>(&input, {1, 2, 3, 4});

  std::vector<tf::Tensor> outputs;
  TF_RETURN_IF_ERROR(
      session->Run({{"A", input}}, {"Matmul"}, /*target nodes*/ {}, &outputs));

  return Status::OK();
}

struct IpuConfigTest : ::testing::Test {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        auto tf_ipu_count,
        xla::poplarplugin::HloPoplarTestBase::GetMaxIpuCount());

    const bool running_on_hw = tf_ipu_count > 0;
    if (!running_on_hw) {
      GTEST_SKIP() << "Skipping IpuConfigTests. They need to run on HW as "
                      "IPU-model can be used without explicit configuration. "
                      "Set TF_IPU_COUNT when running locally.";
    }
  }

  IpuConfig CreateTestConfig(int device_count = 1) {
    IpuConfig ipu_config;

    // We need to set the device type to on demand for running tests on the CI.
    ipu_config.options.set_device_connection_type(
        xla::poplarplugin::IpuDeviceConnectionType::ON_DEMAND);

    while (device_count--) {
      ipu_config.options.add_device_config()->set_auto_count(1);
    }

    return ipu_config;
  }
};

TEST_F(IpuConfigTest, NoDefaultConfig) {
  using ::testing::IsEmpty;

  std::vector<IpuConfig> configs;
  ASSERT_STATUS_OK(GetIpuConfig(configs));
  ASSERT_THAT(configs, IsEmpty());
}

TEST_F(IpuConfigTest, ReturnsSetConfiguration) {
  using ::testing::ElementsAre;
  using ::testing::SizeIs;

  IpuConfig ipu_config = CreateTestConfig(/*device_count*/ 2);
  ASSERT_STATUS_OK(ConfigureIpuSystem(ipu_config));

  std::vector<IpuConfig> configs;
  ASSERT_STATUS_OK(GetIpuConfig(configs));

  ASSERT_THAT(configs, SizeIs(2));
  ASSERT_THAT(configs, ElementsAre(ipu_config, ipu_config));
}

TEST_F(IpuConfigTest, MustConfigureBeforeUsingIPU) {
  ASSERT_STATUS_BAD(RunSimpleGraph("/device:IPU:0"));

  IpuConfig ipu_config = CreateTestConfig();

  ASSERT_STATUS_OK(ConfigureIpuSystem(ipu_config));
  ASSERT_STATUS_OK(RunSimpleGraph("/device:IPU:0"));
}

TEST_F(IpuConfigTest, ReconfiguringFails) {
  IpuConfig ipu_config = CreateTestConfig();

  ASSERT_STATUS_OK(ConfigureIpuSystem(ipu_config));

  ipu_config.options.add_device_config()->set_auto_count(2);

  ASSERT_STATUS_BAD(ConfigureIpuSystem(ipu_config));
}

}  // namespace
}  // namespace ipu
}  // namespace tensorflow

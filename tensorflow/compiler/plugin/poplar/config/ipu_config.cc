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

#include "tensorflow/compiler/plugin/poplar/config/ipu_config.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace ipu {
namespace {
Status UpdateScopeWithBuilder(tensorflow::Scope& scope,
                              tensorflow::NodeBuilder& builder) {
  tensorflow::Node* node;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &node));
  if (!scope.ok()) {
    return scope.status();
  }
  scope.UpdateStatus(scope.DoShapeInference(node));

  return Status::OK();
}

Status CreateIpuConfigureHardwareOp(tensorflow::Scope&& scope,
                                    const IpuConfig& config) {
  std::string config_attr;
  config.options.SerializeToString(&config_attr);

  const std::string unique_name =
      scope.GetUniqueNameForOp("IpuConfigureHardware");
  auto builder = tensorflow::NodeBuilder(unique_name, "IpuConfigureHardware")
                     .Attr("config", config_attr)
                     .Device("/cpu:0");

  return UpdateScopeWithBuilder(scope, builder);
}

Status CreateIpuGetConfigurationOp(tensorflow::Scope&& scope) {
  const std::string unique_name =
      scope.GetUniqueNameForOp("IpuGetConfiguration");
  auto builder = tensorflow::NodeBuilder(unique_name, "IpuGetConfiguration")
                     .Device("/cpu:0");

  return UpdateScopeWithBuilder(scope, builder);
}

Status CreateSession(tensorflow::Scope& scope,
                     std::unique_ptr<tensorflow::Session>& session) {
  tensorflow::Session* raw_session = nullptr;
  TF_RETURN_IF_ERROR(
      tensorflow::NewSession(/*SessionOptions*/ {}, &raw_session));

  CHECK(raw_session);
  session.reset(raw_session);

  tensorflow::GraphDef graph;
  scope.graph()->ToGraphDef(&graph);
  TF_RETURN_IF_ERROR(session->Create(graph));

  return Status::OK();
}
}  // namespace

IpuConfig::IpuConfig() {
  options.set_creator_id(xla::poplarplugin::IpuOptionsCreator::IPU_UTILS);
}

Status ConfigureIpuSystem(const IpuConfig& config) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  TF_RETURN_IF_ERROR(
      CreateIpuConfigureHardwareOp(root.WithOpName("set_ipu_config"), config));

  std::unique_ptr<tensorflow::Session> session;
  TF_RETURN_IF_ERROR(CreateSession(root, session));
  CHECK(session);

  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run(/*inputs*/ {}, /*outputs*/ {},
                                  {"set_ipu_config"}, &outputs));

  return Status::OK();
}

Status GetIpuConfig(std::vector<IpuConfig>& configs) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  TF_RETURN_IF_ERROR(
      CreateIpuGetConfigurationOp(root.WithOpName("get_ipu_config")));

  std::unique_ptr<tensorflow::Session> session;
  TF_RETURN_IF_ERROR(CreateSession(root, session));
  CHECK(session);

  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run(/*inputs*/ {}, {"get_ipu_config"},
                                  /*target nodes*/ {}, &outputs));

  // Convert string tensor to IpuOptions
  CHECK_EQ(outputs.size(), 1);
  const tensorflow::Tensor& output = outputs.front();
  const auto output_data = output.flat<tstring>();

  for (auto i = 0; i < output.NumElements(); ++i) {
    configs.emplace_back();
    configs.back().options.ParseFromString(output_data(i));
  }

  return Status::OK();
}

}  // namespace ipu
}  // namespace tensorflow

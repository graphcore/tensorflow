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

#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

#include <vector>

namespace xla {
namespace poplarplugin {

std::string GetInputCopyHandle(int64 parameter, int64 index) {
  return tensorflow::strings::Printf("%lld.%lld", parameter, index);
}

std::string GetOutputCopyHandle(int64 output_index, int64 flat_tensor_index) {
  return tensorflow::strings::Printf("out_%lld.%lld", output_index,
                                     flat_tensor_index);
}

InputOutputAliasingMap::InputOutputAliasingMap(const HloModule* module) {
  const auto& inputs = module->entry_computation()->parameter_instructions();
  const auto& argument_input_indices =
      module->config().argument_input_indices();
  const auto& resource_input_indices =
      module->config().resource_input_indices();
  const auto& resource_input_initialized =
      module->config().resource_input_initialized();
  const auto& resource_update_to_input_index =
      module->config().resource_update_to_input_index();
  CHECK_EQ(resource_input_indices.size(), resource_input_initialized.size());

  // An XLA entry computation has a set of input parameters. These map to a
  // combination of the inputs to the _XlaRun TF Op, and the resources which
  // are used by it. However not all inputs to the TF Op are Parameters in the
  // XLA entry computation:
  // * Constants are sinked into the computation.
  // * Uninitialized resources don't have a parameter as they don't have a value
  // yet (they will be outputs only).
  //
  // The `argument_input_indices` stores the indices of arguments in the
  // TF Op excluding constants and resource variables.
  // The `resource_input_indices` stores the indices of arguments in the
  // TF Op which are resource variables.
  // The `resource_input_initialized` stores for an element at index `i` whether
  // the resource input `resource_input_indices[i]` is initialised.
  // The `resource_update_to_input_index` stores the input parameter index of a
  // resource which was updated.
  //
  // Note that there is no strict ordering of inputs (resource and non resource
  // arguments can be interleaved), however for outputs all non-resource outputs
  // are the first N outputs, followed by the resource updates.
  //
  // For example, given following arguments to a TF Op:
  // arg0 - Constant
  // arg1 - Resource
  // arg2 - Resource (not initialised)
  // arg3 - Streamed input
  // arg4 - Resource
  // And outputs:
  // out0 - Non resource output
  // out1 - Resource update of arg1
  // out2 - Resource update of arg2
  //
  // We expect:
  // argument_input_indices = [3]
  // resource_input_indices = [1, 2, 4]
  // resource_input_initialized = [True, False, True]
  // resource_update_to_input_index = [1, 2]
  //
  // This means that the Hlo computation will have three parameters (arg1, arg3,
  // arg4).
  // From this information we build a mapping from the Op arguments to Hlo
  // parameters.
  // In the above example we expect the following:
  // streamed_parameters = [1]
  // resource_parameters = [0, 2]
  // input_index_to_parameter = {1 -> 0, 4 -> 2}.
  std::vector<int32> streamed_parameters;
  std::vector<int32> resource_parameters;
  absl::flat_hash_set<int32> resource_parameters_set;
  absl::flat_hash_map<int32, int32> input_index_to_parameter;

  if (argument_input_indices.empty() && resource_input_indices.empty()) {
    // If the values are not set assume everything is streamed.
    streamed_parameters.resize(inputs.size());
    absl::c_iota(streamed_parameters, 0);
  } else {
    CHECK_EQ(inputs.size(),
             argument_input_indices.size() +
                 absl::c_count(resource_input_initialized, true));
    // Map the inputs to hlo parameters - note that streamed inputs and
    // resources might be interleaved. Only initialized resources are mapped to
    // parameters.
    auto argument_itr = argument_input_indices.begin();
    auto resource_itr = resource_input_indices.begin();
    auto initialized_itr = resource_input_initialized.begin();

    bool argument_end = argument_itr == argument_input_indices.end();
    bool resource_end = resource_itr == resource_input_indices.end();
    int64 next_parameter_idx = 0;
    while (!argument_end || !resource_end) {
      if (resource_end ||
          (!argument_end && !resource_end && *argument_itr < *resource_itr)) {
        CHECK(!argument_end);
        streamed_parameters.push_back(next_parameter_idx++);
        ++argument_itr;
      } else {
        CHECK(!resource_end);
        // Only map an input to a parameter if it is initialized.
        if (*initialized_itr) {
          input_index_to_parameter[*resource_itr] = next_parameter_idx;
          resource_parameters_set.insert(next_parameter_idx);
          resource_parameters.push_back(next_parameter_idx++);
        }
        ++resource_itr;
        ++initialized_itr;
      }

      argument_end = argument_itr == argument_input_indices.end();
      resource_end = resource_itr == resource_input_indices.end();
    }
    CHECK_EQ(inputs.size(),
             streamed_parameters.size() + resource_parameters.size());
  }
  num_streaming_inputs_ = streamed_parameters.size();

  for (uint64 idx = 0; idx < inputs.size(); ++idx) {
    const bool is_resource = resource_parameters_set.contains(idx);
    const InputInfo::Type type = is_resource
                                     ? InputInfo::Type::ResourceNotModified
                                     : InputInfo::Type::StreamedVariable;
    entry_input_infos_.push_back(
        InputInfo(type, inputs[idx]->metadata().op_name(), inputs[idx]->shape(),
                  entry_input_infos_.size()));
  }

  // Get the shapes.
  const auto& root = module->entry_computation()->root_instruction();
  std::vector<Shape> output_shapes;
  std::vector<std::string> output_names;
  if (root->shape().IsTuple()) {
    int64 tuple_index = 0;
    for (auto shape : root->shape().tuple_shapes()) {
      output_shapes.push_back(shape);
      if (root->opcode() == HloOpcode::kTuple) {
        output_names.push_back(
            root->operand(tuple_index)->metadata().op_name());
      } else {
        output_names.push_back(
            absl::StrCat(root->metadata().op_name(), ".", tuple_index));
      }
      tuple_index++;
    }
  } else {
    output_shapes.push_back(root->shape());
    output_names.push_back(root->metadata().op_name());
  }

  const uint64 num_outputs = output_shapes.size();
  const uint64 num_resource_updates = resource_update_to_input_index.size();
  num_streaming_outputs_ = num_outputs - num_resource_updates;

  // XLA outputs are ordered such that non resource variables are first.
  for (uint64 idx = 0; idx < num_outputs; ++idx) {
    if (idx < num_streaming_outputs_) {
      entry_output_infos_.push_back(
          OutputInfo(OutputInfo::Type::StreamedVariable, output_names[idx],
                     output_shapes[idx], entry_output_infos_.size()));
    } else {
      const int32 resource_idx = idx - num_streaming_outputs_;
      const int32 input_index = resource_update_to_input_index[resource_idx];
      // If the input index was mapped to a parameter, then this is a resource
      // update, otherwise this is an initialisation of a resource.
      if (input_index_to_parameter.contains(input_index)) {
        const int32 parameter_idx = input_index_to_parameter[input_index];
        entry_input_infos_[parameter_idx].ChangeToResourceModified(idx);
        entry_output_infos_.push_back(OutputInfo(
            OutputInfo::Type::ResourceModified, output_names[idx],
            output_shapes[idx], parameter_idx, entry_output_infos_.size()));
      } else {
        entry_output_infos_.push_back(
            OutputInfo(OutputInfo::Type::ResourceOutputOnly, output_names[idx],
                       output_shapes[idx], entry_output_infos_.size()));
      }
    }
  }
}

const std::vector<InputOutputAliasingMap::InputInfo>&
InputOutputAliasingMap::GetEntryInputInfos() const {
  return entry_input_infos_;
}

const std::vector<InputOutputAliasingMap::OutputInfo>&
InputOutputAliasingMap::GetEntryOutputInfos() const {
  return entry_output_infos_;
}

const uint64& InputOutputAliasingMap::GetNumStreamingInputs() const {
  return num_streaming_inputs_;
}

const uint64& InputOutputAliasingMap::GetNumStreamingOutputs() const {
  return num_streaming_outputs_;
}

const bool InputOutputAliasingMap::InputInfo::IsStreaming() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::StreamedVariable;
}

const std::string& InputOutputAliasingMap::InputInfo::Name() const {
  return name_;
}

const std::string& InputOutputAliasingMap::OutputInfo::Name() const {
  return name_;
}

const Shape& InputOutputAliasingMap::InputInfo::Shape() const { return shape_; }

const Shape& InputOutputAliasingMap::OutputInfo::Shape() const {
  return shape_;
}

const bool InputOutputAliasingMap::InputInfo::IsResource() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::ResourceModified ||
         type_ == InputOutputAliasingMap::InputInfo::Type::ResourceNotModified;
}

const bool InputOutputAliasingMap::InputInfo::IsResourceNotModified() const {
  return type_ == InputOutputAliasingMap::InputInfo::Type::ResourceNotModified;
}

const uint64 InputOutputAliasingMap::InputInfo::GetOutputIndex() const {
  return output_index_;
}

const int64 InputOutputAliasingMap::InputInfo::GetParameterIndex() const {
  return parameter_index_;
}

InputOutputAliasingMap::InputInfo::InputInfo(const Type type,
                                             const std::string& name,
                                             const xla::Shape& shape,
                                             int64 parameter_idx)
    : type_(type),
      output_index_(0),
      name_(name),
      shape_(shape),
      parameter_index_(parameter_idx) {
  int64 index = 0;
  for (auto shape : FlattenedXlaShape(shape)) {
    handles_.push_back(GetInputCopyHandle(parameter_idx, index));
    index++;
  }
}

InputOutputAliasingMap::OutputInfo::OutputInfo(const Type& type,
                                               const std::string& name,
                                               const xla::Shape& shape,
                                               const uint64 input_index,
                                               int64 parameter_idx)
    : type_(type), input_index_(input_index), name_(name), shape_(shape) {
  int64 index = 0;
  for (auto shape : FlattenedXlaShape(shape)) {
    handles_.push_back(GetOutputCopyHandle(parameter_idx, index));
    index++;
  }
}

const std::vector<std::string>& InputOutputAliasingMap::InputInfo::Handles()
    const {
  return handles_;
}

const std::vector<std::string>& InputOutputAliasingMap::OutputInfo::Handles()
    const {
  return handles_;
}

const bool InputOutputAliasingMap::OutputInfo::IsStreaming() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::StreamedVariable;
}

const bool InputOutputAliasingMap::OutputInfo::IsResource() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceModified ||
         type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceOutputOnly;
}

const bool InputOutputAliasingMap::OutputInfo::IsResourceModified() const {
  return type_ == InputOutputAliasingMap::OutputInfo::Type::ResourceModified;
}

const uint64 InputOutputAliasingMap::OutputInfo::GetInputIndex() const {
  return input_index_;
}

std::string InputOutputAliasingMap::ToString() const {
  std::stringstream ss;
  ss << "== Input information ==\n";
  ss << "Num streaming = " << num_streaming_inputs_ << "\n";
  for (size_t i = 0; i < entry_input_infos_.size(); i++) {
    auto& ip = entry_input_infos_[i];
    ss << " " << i << " (" << ip.Name() << "): ";
    if (ip.IsStreaming()) {
      ss << "streaming";
    } else {
      ss << (ip.IsResource() ? "resource" : "");
      if (ip.IsResourceNotModified()) {
        ss << " (not modified)";
      } else {
        ss << ", output index = " << ip.GetOutputIndex();
      }
    }
    ss << ", shape = " << ip.Shape().ToString() << "\n";
  }
  ss << "== Output information ==\n";
  ss << "Num streaming = " << num_streaming_outputs_ << "\n";
  for (size_t i = 0; i < entry_output_infos_.size(); i++) {
    auto& op = entry_output_infos_[i];
    ss << " " << i << " (" << op.Name() << "): ";
    if (op.IsStreaming()) {
      ss << "streaming";
    } else if (op.IsResourceModified()) {
      ss << "modified resource, input index = " << op.GetInputIndex();
    } else {
      ss << "initialized resource";
    }
    ss << ", shape = " << op.Shape().ToString() << "\n";
  }

  return ss.str();
}

}  // namespace poplarplugin
}  // namespace xla

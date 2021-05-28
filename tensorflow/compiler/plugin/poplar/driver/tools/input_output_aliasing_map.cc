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
  const auto& input_mapping = module->config().input_mapping();
  const auto& resource_update_to_input_index =
      module->config().resource_update_to_input_index();

  /*
   * An XLA entry computation has a set of input parameters.  These map to a
   * combination of the inputs to the _XlaRun TF Op, and the resources which
   * are used by it.
   *
   * The `num_arguments` variable stores the total number of arguments in the
   * original _XlaRun operation.  This does not include the resource variables.
   *
   * The `num_resource_inputs` gives the total number of resource variables in
   * the original _XlaRun Op.
   *
   * `input_mapping` contains a map from the XLA computation parameters to the
   * _XlaRun arguments.  The number of entries in the `input_mapping` may be
   * less than the sum of `num_arguments` and `num_resources` when there are
   * uninitialized ResourceVariables, which are not passed to the XLA
   * Computation.
   */

  if (module->config().argument_count() == 0 &&
      module->config().resource_input_count() == 0) {
    // The information is not available.  Assume all inputs and outputs are
    // streamed.
    num_streaming_inputs_ = inputs.size();
  } else {
    num_streaming_inputs_ = module->config().argument_count();
  }

  for (uint64 idx = 0; idx < inputs.size(); ++idx) {
    bool is_resource = (idx >= num_streaming_inputs_);
    const InputInfo::Type type = is_resource
                                     ? InputInfo::Type::ResourceNotModified
                                     : InputInfo::Type::StreamedVariable;
    entry_input_infos_.push_back(
        InputInfo(type, inputs[idx]->metadata().op_name(), inputs[idx]->shape(),
                  entry_input_infos_.size()));
  }

  /*
   * The `resource_update_to_input_index` is a map from the computation output
   * to a _XlaRun input.
   */
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

  uint64 num_resource_updates = resource_update_to_input_index.size();
  num_streaming_outputs_ = num_outputs - num_resource_updates;

  for (uint64 idx = 0; idx < num_outputs; ++idx) {
    if (idx < num_streaming_outputs_) {
      entry_output_infos_.push_back(
          OutputInfo(OutputInfo::Type::StreamedVariable, output_names[idx],
                     output_shapes[idx], entry_output_infos_.size()));
    } else {
      const uint64 resource_idx = idx - num_streaming_outputs_;
      const uint64 input_index = resource_update_to_input_index[resource_idx];
      auto input_map_it = absl::c_find(input_mapping, input_index);
      if (input_map_it != input_mapping.end()) {
        uint64 parameter_index =
            std::distance(input_mapping.begin(), input_map_it);

        if (num_streaming_inputs_ <= parameter_index) {
          entry_output_infos_.push_back(OutputInfo(
              OutputInfo::Type::ResourceModified, output_names[idx],
              output_shapes[idx], parameter_index, entry_output_infos_.size()));
          entry_input_infos_[parameter_index].ChangeToResourceModified(idx);
        } else {
          entry_output_infos_.push_back(OutputInfo(
              OutputInfo::Type::ResourceOutputOnly, output_names[idx],
              output_shapes[idx], entry_output_infos_.size()));
        }
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

InputOutputAliasingMap::InputInfo::InputInfo(const Type type,
                                             const std::string& name,
                                             const xla::Shape& shape,
                                             int64 parameter_idx)
    : type_(type), output_index_(0), name_(name), shape_(shape) {
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

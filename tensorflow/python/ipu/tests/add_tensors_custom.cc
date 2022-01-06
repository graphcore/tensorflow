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

#include <iostream>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

extern "C" {
int32_t custom_op_api_level = 5;
}

extern "C" void Callback(const std::vector<void*>& data,
                         const std::vector<uint32_t>& number_of_elements,
                         std::vector<void*>& outputs,
                         const std::string& attributes,
                         const std::string& name) {
  float* input1 = static_cast<float*>(data[0]);
  float* input2 = static_cast<float*>(data[1]);
  float* output = static_cast<float*>(outputs[0]);
  int number_of_elements_per_tensor = number_of_elements[0];

  for (int i = 0; i < number_of_elements_per_tensor; i++) {
    *output = (*input1) + (*input2);
    input1++;
    input2++;
    output++;
  }
}

extern "C" void Callback_grad(const std::vector<void*>& data,
                              const std::vector<uint32_t>& number_of_elements,
                              std::vector<void*>& outputs,
                              const std::string& attributes,
                              const std::string& name) {
  float* output = static_cast<float*>(outputs[0]);

  // Dummy return values
  for (int i = 0; i < number_of_elements[0]; i++) {
    *output = 1.0;
    output++;
  }
}

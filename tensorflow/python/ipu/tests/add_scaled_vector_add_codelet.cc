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

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

// NOLINTNEXTLINE
using namespace poplar;
template <typename FPType>
class ScaledVectorAdd : public Vertex {
 public:
  Vector<Input<Vector<FPType>>> x;
  Vector<Input<Vector<FPType>>> y;
  Input<Vector<FPType>> scale;
  Vector<Output<Vector<FPType>>> z;
  bool compute() {
    for (unsigned i = 0; i < x.size(); ++i) {
      for (unsigned j = 0; j != x[i].size(); ++j) {
        z[i][j] = x[i][j] + y[i][j] * scale[0];
      }
    }
    return true;
  }
};
template class ScaledVectorAdd<float>;
template class ScaledVectorAdd<half>;

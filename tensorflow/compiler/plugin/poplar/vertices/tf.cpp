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

#include <cmath>
#include <limits>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

// Simple reductions

#define REDUCTION_ELEMENTWISE(NAME, INIT, EXP)  \
  template <typename T>                         \
  class NAME : public Vertex {                  \
   public:                                      \
    Input<Vector<T>> a;                         \
    Output<Vector<T>> out;                      \
                                                \
    bool compute() {                            \
      T v = (INIT);                             \
                                                \
      for (unsigned i = 0; i < a.size(); ++i) { \
        v = (EXP);                              \
      }                                         \
                                                \
      out[0] = v;                               \
      return true;                              \
    }                                           \
  };                                            \
                                                \
  template class NAME<float>;                   \
  template class NAME<half>;                    \
  template class NAME<int>;

REDUCTION_ELEMENTWISE(ReductionMax, std::numeric_limits<T>::lowest(),
                      ((a[i] > v) ? a[i] : v))
REDUCTION_ELEMENTWISE(ReductionMin, std::numeric_limits<T>::max(),
                      ((a[i] < v) ? a[i] : v))
REDUCTION_ELEMENTWISE(ReductionAdd, 0.0, v + a[i])
REDUCTION_ELEMENTWISE(ReductionMul, 1.0, v* a[i])

#define LOGICAL_REDUCTION_ELEMENTWISE(NAME, INIT, EXP) \
  template <typename T>                                \
  class NAME : public Vertex {                         \
   public:                                             \
    Input<Vector<T>> a;                                \
    Output<Vector<T>> out;                             \
                                                       \
    bool compute() {                                   \
      T v = (INIT);                                    \
                                                       \
      for (unsigned i = 0; i < a.size(); ++i) {        \
        v = (EXP);                                     \
      }                                                \
                                                       \
      out[0] = v;                                      \
      return out[0];                                   \
    }                                                  \
  };                                                   \
                                                       \
  template class NAME<bool>;

LOGICAL_REDUCTION_ELEMENTWISE(ReductionAnd, true, v&& a[i])
LOGICAL_REDUCTION_ELEMENTWISE(ReductionOr, false, v || a[i])

#define WINDOWED_SELECTION(NAME, OP)              \
  template <typename T>                           \
  class NAME : public Vertex {                    \
   public:                                        \
    Input<Vector<T>> a;                           \
    Input<T> b;                                   \
    Output<Vector<T>> out;                        \
    T initval;                                    \
                                                  \
    bool compute() {                              \
      unsigned selected = 0;                      \
      for (unsigned i = 1; i < a.size(); ++i) {   \
        if (a[i] OP a[selected]) {                \
          selected = i;                           \
        }                                         \
      }                                           \
      for (unsigned i = 0; i < out.size(); ++i) { \
        out[i] = (i == selected) ? b : initval;   \
      }                                           \
      return true;                                \
    }                                             \
  };                                              \
                                                  \
  template class NAME<float>;                     \
  template class NAME<half>;                      \
  template class NAME<int>;

WINDOWED_SELECTION(SelectionGe, >=)
WINDOWED_SELECTION(SelectionGt, >)
WINDOWED_SELECTION(SelectionLe, <=)
WINDOWED_SELECTION(SelectionLt, <)

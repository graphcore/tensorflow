/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PRINT_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PRINT_UTIL_H_

#include <string>
#include <type_traits>
#include "absl/strings/str_join.h"

namespace xla {
namespace poplarplugin {

// Formatter that converts to string by calling the name attribute,
// works for pointes as well as references
class NameFormatter {
  template <class T>
  void dispatch(std::string* out, const T* const t, std::true_type) const {
    return absl::AlphaNumFormatter()(out, t->name());
  }
  template <class T>
  void dispatch(std::string* out, const T& t, std::false_type) const {
    return absl::AlphaNumFormatter()(out, t.name());
  }

 public:
  template <class T>
  void operator()(std::string* out, const T& t) const {
    return dispatch(out, t, std::is_pointer<T>());
  }
};

// Formatter that converts to string by calling the ToString attribute,
// works for pointes as well as references
class ToStringFormatter {
  template <class T>
  void dispatch(std::string* out, const T* const t, std::true_type) const {
    return absl::AlphaNumFormatter()(out, t->ToString());
  }
  template <class T>
  void dispatch(std::string* out, const T& t, std::false_type) const {
    return absl::AlphaNumFormatter()(out, t.ToString());
  }

 public:
  template <class T>
  void operator()(std::string* out, const T& t) const {
    return dispatch(out, t, std::is_pointer<T>());
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PRINT_UTIL_H_

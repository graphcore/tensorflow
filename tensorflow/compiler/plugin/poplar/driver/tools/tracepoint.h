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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TRACEPOINT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TRACEPOINT_H_

#include <string>

#include <pvti/pvti.hpp>

namespace xla {

namespace poplarplugin {

struct string_view {
  const char* ptr;
  size_t len;
};

class TensorflowPoplarPluginTracepoint : public pvti::Tracepoint {
  static pvti::TraceChannel traceTensorflow;

 public:
  explicit TensorflowPoplarPluginTracepoint(const std::string& traceLabel)
      : pvti::Tracepoint(&TensorflowPoplarPluginTracepoint::traceTensorflow,
                         traceLabel) {}

  explicit TensorflowPoplarPluginTracepoint(const char* traceLabel)
      : pvti::Tracepoint(&TensorflowPoplarPluginTracepoint::traceTensorflow,
                         traceLabel) {}

  explicit TensorflowPoplarPluginTracepoint(const string_view traceLabel)
      : pvti::Tracepoint(&TensorflowPoplarPluginTracepoint::traceTensorflow,
                         traceLabel.ptr, traceLabel.len) {}

  ~TensorflowPoplarPluginTracepoint() = default;
};

#if __cplusplus >= 201402L
// If C++ 14 or greater
constexpr string_view format_pretty_function(const char* s) {
  // First find the opening brackets for the arguments
  char const* b = s;
  while (*b != '(' && *b != '\0') {
    b++;
  }

  // Search backwards for the first space
  char const* c = b;
  while (*c != ' ' && c != s) {
    c--;
  }

  // c can equal s if the function has no return type i.e. constructors.
  if (c == s) {
    return {s, static_cast<size_t>(b - s)};
  } else {
    // +1 as c points to the ' '
    return {c + 1, static_cast<size_t>(b - (c + 1))};
  }
}
#elif __cplusplus >= 201103L
// If C++ 11

// The following const expr functions will attempt to find the first '(' and
// then search backwards for the first ' '. Due to the limitations of const expr
// functions in C++, these functions are recursive.

constexpr size_t find_arg_offset(const char* s, size_t i = 0) {
  return (s[i] != '(' && s[i] != '\0') ? find_arg_offset(s, i + 1) : i;
}

constexpr size_t find_name_offset_backwards(const char* s, size_t e,
                                            size_t i = 0) {
  return (s[e - i] != ' ' && i != e) ? find_name_offset_backwards(s, e, i + 1)
                                     : e - i;
}

constexpr string_view format_pretty_function(const char* s) {
  return {&s[find_name_offset_backwards(s, find_arg_offset(s))],
          static_cast<size_t>(find_arg_offset(s) - find_name_offset_backwards(
                                                       s, find_arg_offset(s)))};
}
#else
#error "require C++11 or greater for trace point function name mangaling"
#endif

#define TENSORFLOW_TRACEPOINT()          \
  TensorflowPoplarPluginTracepoint __pt( \
      format_pretty_function(__PRETTY_FUNCTION__))
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_TRACEPOINT_H_

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

#ifndef __vertex_templates_hpp__
#define __vertex_templates_hpp__
#include <poplar/Type.hpp>
#include <string>

inline std::string templateVertexParams(bool first) {
  if (first)
    return "<>";
  else
    return ">";
}

template <typename... Args>
inline std::string templateVertexParams(bool first, const std::string& val,
                                        Args... args);

template <typename... Args>
inline std::string templateVertexParams(bool first, const char* val,
                                        Args... args);

template <typename... Args>
inline std::string templateVertexParams(bool first, const poplar::Type& type,
                                        Args... args);

template <typename T, typename... Args>
inline std::string templateVertexParams(bool first, const T& val,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += std::to_string(val) + templateVertexParams(false, args...);
  return p;
}

template <typename... Args>
inline std::string templateVertexParams(bool first, const poplar::Type& type,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += type.toString() + templateVertexParams(false, args...);
  return p;
}

template <typename... Args>
inline std::string templateVertexParams(bool first, const std::string& val,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, args...);
  return p;
}

template <typename... Args>
inline std::string templateVertexParams(bool first, const char* val,
                                        Args... args) {
  std::string p = first ? "<" : ",";
  p += val + templateVertexParams(false, args...);
  return p;
}

template <typename... Args>
inline std::string templateVertex(const std::string& name, Args... args) {
  return name + templateVertexParams(true, args...);
}

#endif

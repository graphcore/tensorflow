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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_COMMAND_LINE_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_COMMAND_LINE_UTILS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace ipu {

bool CreateDirIfNeeded(const std::string& dir);
std::string SecondsToTimeString(int64_t seconds);

struct BinaryFiles {
  bool Parse(absl::string_view text, std::string* error);
  std::string Unparse() const;

  std::vector<std::string> filenames;
};

struct CheckpointFile {
  bool Parse(absl::string_view text, std::string* error);
  std::string Unparse() const;
  std::string filename;
};

struct MetadataFile {
  bool Parse(absl::string_view text, std::string* error);
  std::string Unparse() const;
  std::string filename;
};

// These functions need to be static inline in the header for abseil to find
// them.
static std::string AbslUnparseFlag(ipu::BinaryFiles f) { return f.Unparse(); }

static std::string AbslUnparseFlag(ipu::CheckpointFile f) {
  return f.Unparse();
}

static std::string AbslUnparseFlag(ipu::MetadataFile f) { return f.Unparse(); }

static bool AbslParseFlag(absl::string_view text, ipu::MetadataFile* f,
                          std::string* error) {
  return f->Parse(text, error);
}

static bool AbslParseFlag(absl::string_view text, ipu::BinaryFiles* f,
                          std::string* error) {
  return f->Parse(text, error);
}

static bool AbslParseFlag(absl::string_view text, ipu::CheckpointFile* f,
                          std::string* error) {
  return f->Parse(text, error);
}

}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_COMMAND_LINE_UTILS_H_

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
#include "ipu/poplar_command_line_utils.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage_config.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "ipu/poplar_executable_data.h"

namespace {
std::string FileExtension(const std::string& filename,
                          bool no_extension_allowed) {
  size_t dot_pos = filename.rfind(".");
  if (dot_pos == std::string::npos) {
    ERROR_ON_MSG(!no_extension_allowed,
                 "Invalid filename '" << filename << "': no extension");
    return "";
  }
  return filename.substr(dot_pos + 1);
}

std::vector<std::string> ListFiles(const std::string& folder,
                                   std::string* error) {
  std::vector<std::string> files;
  DIR* dirp = opendir(folder.c_str());
  if (dirp == NULL) {
    *error = absl::StrCat("Can't open folder '", folder, "'");
    return {};
  }
  struct dirent* dp;
  while ((dp = readdir(dirp)) != NULL) {
    files.push_back(dp->d_name);
  }
  closedir(dirp);
  return files;
}

bool FileExists(const std::string& filename) {
  return std::ifstream(filename).is_open();
}

bool IsDir(const std::string& path) {
  DIR* dirp = opendir(path.c_str());
  if (dirp) {
    closedir(dirp);
  }
  return dirp != NULL;
}

}  // namespace

namespace ipu {

bool CreateDirIfNeeded(const std::string& dir) {
  DIR* dp = opendir(dir.c_str());
  if (dp == NULL) {
    if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
      return false;
    }
  } else {
    closedir(dp);
  }
  return true;
}

std::string SecondsToTimeString(int64_t sec) {
  int64_t hours = sec / 3600;
  int64_t minutes = (sec - hours * 3600) / 60;
  sec -= hours * 3600 + minutes * 60;
  std::stringstream ss;
  if (hours > 0) {
    ss << hours << "h ";
  }
  if (minutes > 0) {
    ss << minutes << "m ";
  }
  ss << sec << "s";
  return ss.str();
}

std::string BinaryFiles::Unparse() const {
  return absl::UnparseFlag(filenames);
}

std::string CheckpointFile::Unparse() const {
  return absl::UnparseFlag(filename);
}

bool BinaryFiles::Parse(absl::string_view text, std::string* error) {
  std::vector<std::string> names;
  if (!absl::ParseFlag(text, &names, error)) {
    return false;
  }
  for (auto name : names) {
    if (IsDir(name)) {
      std::vector<std::string> files = ListFiles(name, error);
      if (!error->empty()) {
        return false;
      }
      for (auto file : files) {
        if (FileExtension(file, true) == "bin" ||
            FileExtension(file, true) == "ipu_bin") {
          filenames.push_back(absl::StrCat(name, "/", file));
        }
      }
    } else {
      if (!FileExists(name)) {
        *error = absl::StrCat("Could not open file '", name, "'.");
        return false;
      }
      filenames.push_back(name);
    }
  }
  return true;
}

bool CheckpointFile::Parse(absl::string_view text, std::string* error) {
  if (!absl::ParseFlag(text, &filename, error)) {
    return false;
  }
  if (IsDir(filename)) {
    ERROR_ON(!error->empty());
    filename = absl::StrCat(filename, "/ckpt.json");
  }

  if (!FileExists(filename)) {
    *error = absl::StrCat("Could not open checkpoint file '", filename, "'");
    return false;
  }
  return true;
}

std::string MetadataFile::Unparse() const {
  return absl::UnparseFlag(filename);
}

bool MetadataFile::Parse(absl::string_view text, std::string* error) {
  std::string name;
  if (!absl::ParseFlag(text, &name, error)) {
    return false;
  }
  if (IsDir(name)) {
    std::vector<std::string> files = ListFiles(name, error);
    if (!error->empty()) {
      return false;
    }
    for (auto file : files) {
      if (FileExtension(file, true) == "json") {
        if (!filename.empty()) {
          *error = absl::StrCat(
              "There is more than one json file in the directory '", name,
              "' please specify which one you want to use");
          return false;
        }
        filename = absl::StrCat(name, "/", file);
      }
    }
  } else {
    if (!FileExists(name)) {
      *error = absl::StrCat("Could not open file '", name, "'.");
      return false;
    }
    filename = name;
  }
  return true;
}

}  // namespace ipu

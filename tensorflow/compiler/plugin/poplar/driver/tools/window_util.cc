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

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/window_util.h"

namespace xla {
namespace poplarplugin {

xla::StatusOr<xla::Window> MakeWindow(
    const xla::Shape& input_shape, absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides, xla::Padding xla_padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation) {
  TF_RETURN_IF_ERROR(
      xla::ValidatePaddingValues(xla::AsInt64Slice(input_shape.dimensions()),
                                 window_dimensions, window_strides));

  std::vector<std::pair<int64_t, int64_t>> padding =
      xla::MakePadding(xla::AsInt64Slice(input_shape.dimensions()),
                       window_dimensions, window_strides, xla_padding);

  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return Status::OK();
    } else {
      return xla::InvalidArgument(
          "%s", absl::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name,
                    "\nNumber of window dimensions: ", window_dimensions.size(),
                    "\nNumber of ", x_name, ": ", x, "\n"));
    }
  };
  TF_RETURN_IF_ERROR(verify_size(window_strides.size(), "window strides"));
  TF_RETURN_IF_ERROR(verify_size(padding.size(), "padding entries"));
  TF_RETURN_IF_ERROR(verify_size(lhs_dilation.size(), "lhs dilation factors"));
  TF_RETURN_IF_ERROR(verify_size(rhs_dilation.size(), "rhs dilation factors"));

  xla::Window window;
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return window;
}

std::string WindowDimensionToString(const xla::WindowDimension& window_dim) {
  std::ostringstream oss;
  oss << "{" << std::endl;
  oss << "\tsize: " << window_dim.size() << std::endl;
  oss << "\tstride: " << window_dim.stride() << std::endl;
  oss << "\tpadding_low: " << window_dim.padding_low() << std::endl;
  oss << "\tpadding_high: " << window_dim.padding_high() << std::endl;
  oss << "\twindow_dilation: " << window_dim.window_dilation() << std::endl;
  oss << "\tbase_dilation: " << window_dim.base_dilation() << std::endl;
  oss << "\twindow_reversal: " << window_dim.window_reversal() << std::endl;
  oss << "}";

  return oss.str();
}

std::string WindowToString(const xla::Window& window) {
  const auto n = window.dimensions_size();
  std::ostringstream oss;
  oss << "Window has " << n << " dimensions." << std::endl;
  for (int i = 0; i < n; i++) {
    oss << "Window Dimension " << i << " configuration:" << std::endl;
    oss << WindowDimensionToString(window.dimensions(i));

    if (i < (n - 1)) {
      oss << std::endl;
    }
  }
  return oss.str();
}

}  // namespace poplarplugin
}  // namespace xla

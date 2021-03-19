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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_POPFLOAT_CAST_TO_GFLOAT_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_POPFLOAT_CAST_TO_GFLOAT_UTILS_H_

#include "tensorflow/compiler/plugin/poplar/kernels/popfloat/gfloat_config_utils.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include <popfloat/experimental/CastToGfloat.hpp>

namespace gfloatutils {

// Convert FPConfig::RoundMode to popfloat's GfloatRoundType
popfloat::experimental::RoundType GetPopfloatRoundModeType(
    FPConfig::RoundMode round_mode, const bool quantised_fp32,
    unsigned sr_bits);

// Convert GFConfig::GfloatFormat to popfloat's GfloatFormatType
popfloat::experimental::FormatType GetPopfloatFormatType(
    GFConfig::GfloatFormat gf_format);

// Convert SRConfig::Density to popfloat's
// GfloatSRDensityType
popfloat::experimental::SRDensityType GetPopfloatSRDensityType(
    SRConfig::Density noise_density);

popfloat::experimental::GfloatCast::CastConfig CreateCastNativeToGfloatConfig(
    PopfloatCastConfig gf_cast_config, poplar::Type calc_type,
    poplar::Type out_type);

popfloat::experimental::GfloatCast::RoundConfig GetPopfloatGfloatRoundConfig(
    PopfloatCastConfig cast_config);
}  // namespace gfloatutils

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_POPFLOAT_CAST_TO_GFLOAT_UTILS_H_

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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popfloat/gfloat_ops_utils.h"

#include <stdlib.h>

const unsigned fp32_man_size = 23;
const unsigned fp16_man_size = 10;

namespace gfloatutils {
// Convert FPConfig::RoundMode to popfloat's GfloatRoundType
popfloat::experimental::RoundType GetPopfloatRoundModeType(
    FPConfig::RoundMode round_mode, const bool quantised_fp32,
    unsigned sr_bits) {
  switch (round_mode) {
    case FPConfig_RoundMode_RZ:
      return popfloat::experimental::RoundType::RZ;
    case FPConfig_RoundMode_RN:
      return popfloat::experimental::RoundType::RN;
    case FPConfig_RoundMode_RA:
      return popfloat::experimental::RoundType::RA;
    case FPConfig_RoundMode_RU:
      return popfloat::experimental::RoundType::RU;
    case FPConfig_RoundMode_RD:
      return popfloat::experimental::RoundType::RD;
    case FPConfig_RoundMode_SR:
      if (quantised_fp32) {
        return (sr_bits < fp32_man_size)
                   ? popfloat::experimental::RoundType::SR
                   : popfloat::experimental::RoundType::SX;
      } else {
        return (sr_bits < fp16_man_size)
                   ? popfloat::experimental::RoundType::SR
                   : popfloat::experimental::RoundType::SX;
      }
    case FPConfig_RoundMode_INVALID:
      return popfloat::experimental::RoundType::INV;
  }
  LOG(FATAL) << "Unhandled RoundMode: " << round_mode;
}

// Convert GFConfig::GfloatFormat to popfloat's GfloatFormatType
popfloat::experimental::FormatType GetPopfloatFormatType(
    GFConfig::GfloatFormat gfloat_format) {
  switch (gfloat_format) {
    case GFConfig_GfloatFormat_ieeeFp16:
      return popfloat::experimental::FormatType::IEEE_FP16;
    case GFConfig_GfloatFormat_quantisedFp32:
      return popfloat::experimental::FormatType::QUANTISED_FP32;
    case GFConfig_GfloatFormat_quantisedFp16:
      return popfloat::experimental::FormatType::QUANTISED_FP16;
    case GFConfig_GfloatFormat_minNormAlignGf8:
      return popfloat::experimental::FormatType::MIN_NORM_ALIGN_GF8;
    case GFConfig_GfloatFormat_oneFiveTwoGf8:
      return popfloat::experimental::FormatType::ONE_FIVE_TWO_GF8;
    case GFConfig_GfloatFormat_maxNormAlignGf8:
      return popfloat::experimental::FormatType::MAX_NORM_ALIGN_GF8;
    case GFConfig_GfloatFormat_bfloat16:
      return popfloat::experimental::FormatType::BFLOAT16;
    case GFConfig_GfloatFormat_noDenormGf16:
      return popfloat::experimental::FormatType::NO_DENORM_GF16;
    case GFConfig_GfloatFormat_enDenormGf16:
      return popfloat::experimental::FormatType::ENABLE_DENORM_GF16;
    case GFConfig_GfloatFormat_Invalid:
      return popfloat::experimental::FormatType::INVALID_FORMAT;
  }
  LOG(FATAL) << "Unhandled GfloatFormat : " << gfloat_format;
}

// Convert SRConfig::Density to popfloat's
// GfloatSRDensityType
popfloat::experimental::SRDensityType GetPopfloatSRDensityType(
    SRConfig::Density noise_density) {
  switch (noise_density) {
    case SRConfig_Density_Uniform:
      return popfloat::experimental::SRDensityType::UNIFORM;
    case SRConfig_Density_Normal:
      return popfloat::experimental::SRDensityType::NORMAL;
    case SRConfig_Density_TruncatedNormal:
      return popfloat::experimental::SRDensityType::TRUNCATED_NORMAL;
    case SRConfig_Density_Bernoulli:
      return popfloat::experimental::SRDensityType::BERNOULLI;
    case SRConfig_Density_TruncatedLogistic:
      return popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC;
    case SRConfig_Density_Logistic:
      return popfloat::experimental::SRDensityType::LOGISTIC;
    case SRConfig_Density_Laplace:
      return popfloat::experimental::SRDensityType::LAPLACE;
    case SRConfig_Density_TruncatedLaplace:
      return popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE;
    case SRConfig_Density_LogitNormal:
      return popfloat::experimental::SRDensityType::LOGIT_NORMAL;
    case SRConfig_Density_TruncatedLogitNormal:
      return popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL;
    case SRConfig_Density_Invalid:
      return popfloat::experimental::SRDensityType::INVALID;
  }
  LOG(FATAL) << "Unhandled Density type: " << noise_density;
}
popfloat::experimental::GfloatCast::RoundConfig GetPopfloatGfloatRoundConfig(
    PopfloatCastConfig cast_config, poplar::Type calc_type) {
  auto round_mode = GetPopfloatRoundModeType(
      cast_config.fp_config().round_mode(), (calc_type == poplar::FLOAT),
      cast_config.sr_config().sr_bits());

  auto sr_density =
      GetPopfloatSRDensityType(cast_config.sr_config().sr_density());

  return popfloat::experimental::GfloatCast::RoundConfig(
      round_mode, cast_config.sr_config().sr_bits(), calc_type, sr_density,
      cast_config.sr_config().sr_norm_offset(),
      cast_config.sr_config().sr_norm_scale(),
      cast_config.sr_config().sr_norm_min(),
      cast_config.sr_config().sr_norm_max(),
      cast_config.sr_config().sr_bernoulli_prob());
}

popfloat::experimental::GfloatCast::CastConfig CreateCastNativeToGfloatConfig(
    PopfloatCastConfig cast_config, poplar::Type calc_type,
    poplar::Type out_type) {
  auto gfloat_format =
      GetPopfloatFormatType(cast_config.gf_config().gfloat_format());

  auto round_cfg = GetPopfloatGfloatRoundConfig(cast_config, calc_type);

  return popfloat::experimental::GfloatCast::CastConfig::createCastNativeToGF(
      gfloat_format, calc_type, out_type, round_cfg,
      cast_config.fp_config().enable_nanoo());
}
}  // namespace gfloatutils

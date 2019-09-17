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
experimental::popfloat::RoundType GetPopfloatRoundModeType(
    FPConfig::RoundMode round_mode, const bool quantised_fp32,
    unsigned sr_bits) {
  switch (round_mode) {
    case FPConfig_RoundMode_RZ:
      return experimental::popfloat::RoundType::RZ;
    case FPConfig_RoundMode_RN:
      return experimental::popfloat::RoundType::RN;
    case FPConfig_RoundMode_RA:
      return experimental::popfloat::RoundType::RA;
    case FPConfig_RoundMode_RU:
      return experimental::popfloat::RoundType::RU;
    case FPConfig_RoundMode_RD:
      return experimental::popfloat::RoundType::RD;
    case FPConfig_RoundMode_SR:
      if (quantised_fp32) {
        return (sr_bits < fp32_man_size)
                   ? experimental::popfloat::RoundType::SR
                   : experimental::popfloat::RoundType::SX;
      } else {
        return (sr_bits < fp16_man_size)
                   ? experimental::popfloat::RoundType::SR
                   : experimental::popfloat::RoundType::SX;
      }
    case FPConfig_RoundMode_INVALID:
      return experimental::popfloat::RoundType::INV;
  }
}

// Convert GFConfig::GfloatFormat to popfloat's GfloatFormatType
experimental::popfloat::FormatType GetPopfloatFormatType(
    GFConfig::GfloatFormat gfloat_format) {
  switch (gfloat_format) {
    case GFConfig_GfloatFormat_ieeeFp16:
      return experimental::popfloat::FormatType::IEEE_FP16;
    case GFConfig_GfloatFormat_quantisedFp32:
      return experimental::popfloat::FormatType::QUANTISED_FP32;
    case GFConfig_GfloatFormat_quantisedFp16:
      return experimental::popfloat::FormatType::QUANTISED_FP16;
    case GFConfig_GfloatFormat_minNormAlignGf8:
      return experimental::popfloat::FormatType::MIN_NORM_ALIGN_GF8;
    case GFConfig_GfloatFormat_oneFiveTwoGf8:
      return experimental::popfloat::FormatType::ONE_FIVE_TWO_GF8;
    case GFConfig_GfloatFormat_maxNormAlignGf8:
      return experimental::popfloat::FormatType::MAX_NORM_ALIGN_GF8;
    case GFConfig_GfloatFormat_bfloat16:
      return experimental::popfloat::FormatType::BFLOAT16;
    case GFConfig_GfloatFormat_noDenormGf16:
      return experimental::popfloat::FormatType::NO_DENORM_GF16;
    case GFConfig_GfloatFormat_enDenormGf16:
      return experimental::popfloat::FormatType::ENABLE_DENORM_GF16;
    case GFConfig_GfloatFormat_Invalid:
      return experimental::popfloat::FormatType::INVALID_FORMAT;
  }
}

// Convert SRConfig::Density to popfloat's
// GfloatSRDensityType
experimental::popfloat::SRDensityType GetPopfloatSRDensityType(
    SRConfig::Density noise_density) {
  switch (noise_density) {
    case SRConfig_Density_Uniform:
      return experimental::popfloat::SRDensityType::UNIFORM;
    case SRConfig_Density_Normal:
      return experimental::popfloat::SRDensityType::NORMAL;
    case SRConfig_Density_TruncatedNormal:
      return experimental::popfloat::SRDensityType::TRUNCATED_NORMAL;
    case SRConfig_Density_Bernoulli:
      return experimental::popfloat::SRDensityType::BERNOULLI;
    case SRConfig_Density_TruncatedLogistic:
      return experimental::popfloat::SRDensityType::TRUNCATED_LOGISTIC;
    case SRConfig_Density_Logistic:
      return experimental::popfloat::SRDensityType::LOGISTIC;
    case SRConfig_Density_Laplace:
      return experimental::popfloat::SRDensityType::LAPLACE;
    case SRConfig_Density_TruncatedLaplace:
      return experimental::popfloat::SRDensityType::TRUNCATED_LAPLACE;
    case SRConfig_Density_LogitNormal:
      return experimental::popfloat::SRDensityType::LOGIT_NORMAL;
    case SRConfig_Density_TruncatedLogitNormal:
      return experimental::popfloat::SRDensityType::TRUNCATED_LOGIT_NORMAL;
    case SRConfig_Density_Invalid:
      return experimental::popfloat::SRDensityType::INVALID;
  }
}
experimental::popfloat::GfloatCast::RoundConfig GetPopfloatGfloatRoundConfig(
    PopfloatCastConfig cast_config, poplar::Type calc_type) {
  auto round_mode = GetPopfloatRoundModeType(
      cast_config.fp_config().round_mode(), (calc_type == poplar::FLOAT),
      cast_config.sr_config().sr_bits());

  auto sr_density =
      GetPopfloatSRDensityType(cast_config.sr_config().sr_density());

  return experimental::popfloat::GfloatCast::RoundConfig(
      round_mode, cast_config.sr_config().sr_bits(), calc_type, sr_density,
      cast_config.sr_config().sr_norm_offset(),
      cast_config.sr_config().sr_norm_scale(),
      cast_config.sr_config().sr_norm_min(),
      cast_config.sr_config().sr_norm_max(),
      cast_config.sr_config().sr_bernoulli_prob());
}

experimental::popfloat::GfloatCast::CastConfig CreateCastNativeToGfloatConfig(
    PopfloatCastConfig cast_config, poplar::Type calc_type,
    poplar::Type out_type) {
  auto gfloat_format =
      GetPopfloatFormatType(cast_config.gf_config().gfloat_format());

  auto round_cfg = GetPopfloatGfloatRoundConfig(cast_config, calc_type);

  return experimental::popfloat::GfloatCast::CastConfig::createCastNativeToGF(
      gfloat_format, calc_type, out_type, round_cfg,
      cast_config.fp_config().enable_nanoo());
}
}  // namespace gfloatutils

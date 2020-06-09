# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Popfloat generic-float operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.ops import gen_popfloat_ops
from tensorflow.python.framework import dtypes

import numpy as np

__all__ = ["GenericFloat"]

cast_to_gfloat16_param_size = 29
cast_to_gfloat32_param_size = 29


class GenericFloat:
  # pylint:disable=line-too-long
  """XLA compatible.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gfloat       = GenericFloat(man, exp, ...)
      castOutput   = gfloat.cast_native_to_gfloat(input, round_mode='RN',...)
      ...
      unpackOutput = gfloat.cast_gfloat_to_native(castOutput)

  """

  # pylint:enable=line-too-long
  # _saveable_cls = popfloat_gfloat_ops.PopfloatGfloatSaveable

  def __init__(self,
               man=10,
               exp=5,
               bias=15,
               en_denorm=True,
               en_inf=True,
               calc_type='auto',
               name=None):
    """Creates a GenericFloat quantisation engine.
    Args:
      man: number of mantissa bits
      exp: number of exponent bits
      bias: exponent bias
      en_denorm: enable/disable denorms
      en_inf: allow 1 exponent value to be used for INF/NAN
      calc_type: type used for calculation
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking __call__().
    """
    self._gfloat_calc_spec_dtype = calc_type
    self._man = man
    self._exp = exp
    self._bias = bias
    self._en_denorm = en_denorm
    self._en_inf = en_inf and (exp > 0)
    self._packed_bits = self._exp + self._man
    self._name = name
    # Initialise to None. Will be set after set_param()
    self._gfloat_calc_dtype = None
    self._gfloat_storage_dtype = None
    self._gfloat_native_dtype = None
    self._gfloat_native_dtype = None
    self._cast_param_size = None
    self._gfloat_as_int = None
    self._is_set = None
    self._format = "Invalid"
    self._cast_to_gfloat_params = np.empty([1], np.int32)
    self.set_param()

  def calc_quantized_fp16_params(self):
    self._cast_param_size = cast_to_gfloat16_param_size
    man_mask = (1 << (10 - self._man)) - 1
    man_mask = man_mask | (man_mask << 16)
    min_norm_align = 1 if (self._exp < 5 or self._en_inf) else 0
    min_norm_exp = 512 << min_norm_align  # minimum normal exponent
    min_exp = (min_norm_exp >> self._man
               ) if self._en_denorm else min_norm_exp  # Smallest value
    min_output = min_exp | (min_exp << 16)
    min_output = min_output | (min_output << 16)
    max_exp = ((1 << self._exp) - 1) - (1 if
                                        (self._en_inf and
                                         (self._exp > 0)) else 0) - self._bias
    max_float = ((max_exp + 127) << 23) | ((1 << 23) - (1 << (23 - self._man)))
    in_scale = (self._bias - 16) if ((self._exp == 5)
                                     and not self._en_inf) else (self._bias -
                                                                 15)
    in_scale_recip = -in_scale
    if in_scale >= -14:
      in_scale_hlf = ((in_scale + 15) << 10) | (((in_scale + 15) << 10) << 16)
    else:
      in_scale_hlf = (1024 >> (-14 - in_scale)) | ((1024 >>
                                                    (-14 - in_scale)) << 16)

    if in_scale_recip > -14:
      in_scale_recip_hlf = ((in_scale_recip + 15) << 10) | ((
          (in_scale_recip + 15) << 10) << 16)
    else:
      in_scale_recip_hlf = (1024 >> (-14 - in_scale_recip)) | (
          (1024 >> (-14 - in_scale_recip)) << 16)
    if max_exp >= -14:
      max_half_in = ((max_exp + 15) << 10) | ((1 << 10) - (1 <<
                                                           (10 - self._man)))
    else:
      max_half_in = ((1 << 11) - (1 << (10 - self._man))) >> (-14 - max_exp)
    max_half_in = max_half_in | (max_half_in << 16) | (1 << 15)

    max_exp_out = max_exp + in_scale
    if max_exp_out >= -14:
      max_half_out = ((max_exp_out + 15) << 10) | ((1 << 10) -
                                                   (1 << (10 - self._man)))
    else:
      max_half_out = ((1 << 11) - (1 <<
                                   (10 - self._man))) >> (-14 - max_exp_out)
    max_half_out = max_half_out | (max_half_out << 16) | (1 << 15)

    pwr_m_man_10 = -(self._man + 10)
    if pwr_m_man_10 >= -14:
      pwr_m_man_10 = (pwr_m_man_10 + 15) << 10
    else:
      pwr_m_man_10 = (1 << 10) >> (-14 - pwr_m_man_10)
    pwr_p_10_m10 = ((10 + 15) << 10) | (((-10 + 15) << 10) << 16)

    fp8_sgn_mask = (1 << 7) | (1 << 15) | (1 << 23) | (1 << 31)
    man_exp = (127 << (10 - self._man)) | (1 << 15)
    man_exp = man_exp | (man_exp << 16)

    self._cast_to_gfloat_params = np.resize(self._cast_to_gfloat_params,
                                            [self._cast_param_size])
    self._cast_to_gfloat_params[0] = (31 << 10) | (31 << 26)  # F16 EXP MASK
    self._cast_to_gfloat_params[1] = (31 << 10) | (31 << 26)  # F16 EXP MASK
    self._cast_to_gfloat_params[2] = (1 << 31) | (1 << 15)  # F16 SIGN MASK
    self._cast_to_gfloat_params[3] = (1 << 31) | (1 << 15)  # F16 SIGN MASK
    self._cast_to_gfloat_params[4] = 0x7ECE | (0x7ECE << 16
                                               )  # F16 QNAN [31894]
    self._cast_to_gfloat_params[5] = 0x7ECE | (0x7ECE << 16)  # F16 QNAN
    self._cast_to_gfloat_params[6] = ~man_mask  # MAN MASK
    self._cast_to_gfloat_params[7] = ~man_mask  # MAN MASK
    self._cast_to_gfloat_params[8] = min_output  # MIN DENORM
    self._cast_to_gfloat_params[9] = min_output  # MIN DENORM
    self._cast_to_gfloat_params[10] = max_float | (1 << 31
                                                   )  # Smallest negative float
    self._cast_to_gfloat_params[11] = max_float  # largest positive float
    self._cast_to_gfloat_params[12] = (
        in_scale + 127) << 23  # Alignment scaling for float inputs
    self._cast_to_gfloat_params[
        13] = in_scale_hlf  # Alignment scaling for half inputs
    self._cast_to_gfloat_params[14] = (
        in_scale_recip +
        127) << 23  # Alignment scaling reciprocal for float inputs
    self._cast_to_gfloat_params[
        15] = in_scale_recip_hlf  # Alignment scaling reciprocal for half inputs
    self._cast_to_gfloat_params[16] = man_exp  # FP8 mantissa and exponent mask
    self._cast_to_gfloat_params[17] = man_exp  # FP8 mantissa and exponent mask
    self._cast_to_gfloat_params[18] = (30 << 10) | (30 << 26)  # F16 MAX EXP
    self._cast_to_gfloat_params[19] = (30 << 10) | (30 << 26)  # F16 MAX EXP
    self._cast_to_gfloat_params[20] = max_half_in  # Clamp Input half pair
    self._cast_to_gfloat_params[21] = max_half_out  # Clamp Output half pair
    self._cast_to_gfloat_params[22] = min_output  # MIN OUTPUT
    self._cast_to_gfloat_params[23] = pwr_m_man_10  # 2^(-(man+10))
    self._cast_to_gfloat_params[24] = (-1 + 15) << 10  # 2^(-1)
    self._cast_to_gfloat_params[25] = pwr_p_10_m10  # 2^(10)
    self._cast_to_gfloat_params[26] = 5 - self._exp  # Pack Shr align
    self._cast_to_gfloat_params[27] = self._exp + (8 - 5)  # Unpack SHR
    self._cast_to_gfloat_params[28] = fp8_sgn_mask  # FP8 Sign mask

  def calc_quantized_fp32_params(self):
    self._cast_param_size = cast_to_gfloat32_param_size

    max_exp = (
        (1 << self._exp) - 1) - (1 if self._en_inf else 0) - self._bias + 127
    max_value = (max_exp << 23) | ((1 << 23) - (1 << (23 - self._man)))
    min_value = (128 - (self._man if self._en_denorm else 0) - self._bias)
    half_min = (min_value - 1) << 23

    exp_mask = ((1 << self._exp) - 1) << (31 - self._exp)
    max_exp = ((1 << self._exp) - 1) - (1 if
                                        (self._en_inf and
                                         (self._exp > 0)) else 0) - self._bias
    max_float = ((max_exp + 127) << 23)
    max_float = max_float | ((1 << 23) - (1 << (23 - self._man)))
    exp_align = ((128 if self._en_denorm else 127) - self._bias) << 23

    man_exp = 0x7FFF << (self._exp + 23 - 15)

    self._cast_to_gfloat_params = np.resize(self._cast_to_gfloat_params,
                                            [self._cast_param_size])
    self._cast_to_gfloat_params[0] = ~((1 << (23 - self._man)) - 1)  # OUT MASK
    self._cast_to_gfloat_params[1] = ~((1 << (23 - self._man)) - 1)  # OUT MASK
    self._cast_to_gfloat_params[2] = ((1 << 8) - 1) << 23  # EXP MASK
    self._cast_to_gfloat_params[3] = ((1 << 8) - 1) << 23  # EXP MASK
    self._cast_to_gfloat_params[4] = (1 << 31)  # SIGN MASK
    self._cast_to_gfloat_params[5] = (1 << 31)  # SIGN MASK
    self._cast_to_gfloat_params[6] = 0x7FD9C07E  # QNAN MASK
    self._cast_to_gfloat_params[7] = 0x7FD9C07E  # QNAN MASK
    self._cast_to_gfloat_params[8] = ((1 << 9) - 1) << 23  # SIGN EXP MASK
    self._cast_to_gfloat_params[9] = ((1 << 9) - 1) << 23  # SIGN EXP MASK
    self._cast_to_gfloat_params[10] = 1 << 23  # BIT23 MASK
    self._cast_to_gfloat_params[11] = 1 << 23  # BIT23 MASK
    self._cast_to_gfloat_params[12] = max_value | (1 << 31)  # CLAMP OUTPUT
    self._cast_to_gfloat_params[13] = max_value  # CLAMP OUTPUT
    self._cast_to_gfloat_params[14] = half_min  # HALF MIN
    self._cast_to_gfloat_params[15] = half_min  # HALF MIN
    self._cast_to_gfloat_params[16] = man_exp  # Packed bits mask
    self._cast_to_gfloat_params[17] = man_exp  # Packed bits mask
    self._cast_to_gfloat_params[18] = (128 - self._bias) << 23  # MIN NORM
    self._cast_to_gfloat_params[19] = (128 - self._bias) << 23  # MIN NORM
    self._cast_to_gfloat_params[20] = exp_mask  # EXP MASK
    self._cast_to_gfloat_params[21] = exp_mask  # EXP MASK
    self._cast_to_gfloat_params[22] = min_value << 23  # MIN VALUE
    self._cast_to_gfloat_params[23] = 1 if self._en_denorm else 0  # EN DENORM
    self._cast_to_gfloat_params[24] = (128 +
                                       self._bias) << 23  # Pack exponent align
    self._cast_to_gfloat_params[25] = self._exp + 8  # Pack rsh align
    self._cast_to_gfloat_params[26] = exp_align  # Unpack exponent align
    self._cast_to_gfloat_params[27] = 8 + self._exp  # EXP ALIGN 0
    self._cast_to_gfloat_params[28] = 8 - self._exp  # EXP ALIGN 1

  def get_gfloat_storage_dtype(self):
    return self._gfloat_storage_dtype

  def get_gfloat_calc_dtype(self):
    return self._gfloat_calc_dtype

  def get_gfloat_native_dtype(self):
    return self._gfloat_native_dtype

  def cast_native_to_gfloat(self,
                            inputs,
                            en_nanoo=True,
                            round_mode='RN',
                            sr_density='Invalid',
                            sr_bits=23,
                            sr_norm_offset=0.0,
                            sr_norm_scale=1.0,
                            sr_norm_min=-0.5,
                            sr_norm_max=0.5,
                            sr_prob=1.0,
                            storage_type='auto'):
    """Quantize input tensor to generic float format
    Args:
      inputs: a tensor with arbitrary shape.
      params: a tensor with Parameters.
      en_nanoo: generate qNan on overflow if Gfloat's Inf is enabled.
      round_mode: Rounding mode for mantissa quantization.
      sr_density: The distribution of the noise used for SR
      sr_bits: Number of bits used for stochastic rounding.
      sr_norm_mean: Mean used by Normal and Truncated Normal noise.
      sr_norm_var: Standard deviation used by Normal and Truncated Normal noise.
      sr_norm_min: Min value of noise to be added
      sr_norm_max: Max value of noise to be added
      sr_prob: Probability of rounding down for Bernoulli - SR
    Returns:
      a tensor of the same shape as the input
    """

    if en_nanoo and not self._en_inf:
      raise RuntimeWarning(
          "Gfloat format's Infs and Nans are not enabled. Input Nans and Infs ",
          "will be set to 0. The quantization will clip on-overflow")

    if self._is_set is None:
      raise RuntimeError(
          "Cast object properties were modified and object was not reset")
    if storage_type == 'auto':
      out_type = self._gfloat_storage_dtype
    elif storage_type == 'fp32':
      out_type = dtypes.float32
    elif storage_type == 'fp16':
      out_type = dtypes.float16
    else:
      raise RuntimeError("Gfloat storage format not supported")

    return gen_popfloat_ops.cast_native_to_gfloat(
        input=inputs,
        params=ops.convert_to_tensor(self._cast_to_gfloat_params),
        en_nanoo=en_nanoo and self._en_inf,
        round_mode=round_mode,
        sr_density=sr_density,
        sr_bits=sr_bits,
        sr_norm_offset=sr_norm_offset,
        sr_norm_scale=sr_norm_scale,
        sr_norm_min=sr_norm_min,
        sr_norm_max=sr_norm_max,
        sr_prob=sr_prob,
        gfloat_format=self._format,
        calc_type=self._gfloat_calc_dtype,
        out_type=out_type)

  def cast_gfloat_to_native(self, inputs, storage_type='auto'):
    """castFromGfloat an FP8 or FP16 gfloat tensor
    Args:
      inputs: a tensor with arbitrary shape.
    Returns:
      output: a tensor of the same shape as the input
    Raises:
    """
    if not self._is_set:
      raise RuntimeError(
          "Gfloat parameters are not set. Try to reset parameters")

    if self._gfloat_as_int is None or not self._gfloat_as_int:
      return inputs
    if storage_type == 'auto':
      out_type = self._gfloat_native_dtype
    elif storage_type == 'fp32':
      out_type = dtypes.float32
    elif storage_type == 'fp16':
      out_type = dtypes.float16

    outputs = gen_popfloat_ops.cast_gfloat_to_native(
        input=inputs,
        params=ops.convert_to_tensor(self._cast_to_gfloat_params),
        gfloat_format=self._format,
        calc_type=self._gfloat_calc_dtype,
        out_type=out_type)
    return outputs

  # Define quantize function
  def get_cast_native_to_gfloat_fun(self,
                                    en_nanoo=True,
                                    round_mode='RN',
                                    sr_bits=23,
                                    sr_density='Invalid',
                                    sr_norm_offset=0.0,
                                    sr_norm_scale=1.0,
                                    sr_norm_min=-0.5,
                                    sr_norm_max=0.5,
                                    sr_prob=1.0,
                                    storage_type='auto'):
    return lambda x: self.cast_native_to_gfloat(x,
                                                en_nanoo=en_nanoo,
                                                round_mode=round_mode,
                                                sr_bits=sr_bits,
                                                sr_density=sr_density,
                                                sr_norm_offset=sr_norm_offset,
                                                sr_norm_scale=sr_norm_scale,
                                                sr_norm_min=sr_norm_min,
                                                sr_norm_max=sr_norm_max,
                                                sr_prob=sr_prob,
                                                storage_type=storage_type)

  # Define cast from gfloat function
  def get_cast_gfloat_to_native_fun(self, storage_type='auto'):
    return lambda x: self.cast_gfloat_to_native(x, storage_type=storage_type)

  def set_param(self):
    if self._is_set is not None:
      return
    params_struct = self._en_denorm | (self._en_inf << 1)
    self._gfloat_packed_params = self._man | (self._exp << 8) | (
        (self._bias & 255) << 16) | (params_struct << 24)
    self._is_set = True

    self._gfloat_storage_dtype = dtypes.float32
    if self._exp > 5 or self._man > 10 or (self._exp == 5 and self._man == 10):
      self._format = "quantisedFp32"
      self._gfloat_native_dtype = dtypes.float16 if (
          self._exp == 5 and self._man == 10
          and self._en_inf) else dtypes.float32
      if self._exp == 5 and self._man == 10:
        self._gfloat_storage_dtype = dtypes.float16
        self._format = "ieeeFp16"
      elif self._packed_bits < 8:
        self._gfloat_storage_dtype = dtypes.int8
        self._format = "enDenormGf16"
        self._gfloat_as_int = True
      elif self._packed_bits < 16:
        self._gfloat_storage_dtype = dtypes.int16
        self._gfloat_native_dtype = dtypes.float32
        self._gfloat_as_int = True
        if self._exp == 8:
          self._format = "bfloat16"
        elif self._en_denorm:
          self._format = "enDenormGf16"
        else:
          self._format = "noDenormGf16"
      if self._gfloat_calc_spec_dtype == 'auto' or self._gfloat_calc_spec_dtype == 'fp32':
        self._gfloat_calc_dtype = dtypes.float32
      elif self._gfloat_calc_spec_dtype == 'fp16':
        raise RuntimeError("Gfloat format cannot be generated using IEEE FP16")
      else:
        raise RuntimeError("Gfloat calculation type not supported")
    else:
      self._format = "quantisedFp16"
      self._gfloat_storage_dtype = dtypes.float16
      if self._exp == 5 and not self._en_inf:
        self._format = "maxNormAlignGf8"
      self._gfloat_native_dtype = dtypes.float16
      if self._packed_bits < 8:
        self._gfloat_as_int = True
        self._gfloat_storage_dtype = dtypes.int8
        self._gfloat_native_dtype = dtypes.float16
        if self._exp == 5:
          self._format = "oneFiveTwoGf8" if self._en_inf else "maxNormAlignGf8"
        else:
          self._format = "minNormAlignGf8"

      if self._gfloat_calc_spec_dtype == 'auto' or self._gfloat_calc_spec_dtype == 'fp16':
        self._gfloat_calc_dtype = dtypes.float16
      elif self._gfloat_calc_spec_dtype == 'fp32':
        self._gfloat_calc_dtype = dtypes.float32
      else:
        raise RuntimeError("Gfloat calculation type not supported")

    if self._gfloat_calc_dtype == dtypes.float32:
      self.calc_quantized_fp32_params()
    elif self._gfloat_calc_dtype == dtypes.float16:
      self.calc_quantized_fp16_params()

  def get_cast_to_gfloat_params(self):
    return self._cast_to_gfloat_params

  def get_gfloat_cast_param_size(self):
    return self._cast_param_size

  def reset(self):
    self._is_set = None
    self.set_param()

  def get_gfloat_packed_params(self):
    return self._gfloat_packed_params

  def get_mantissa_size(self):
    return self._man

  def get_exponent_size(self):
    return self._exp

  def get_exponent_bias(self):
    return self._bias

  def get_en_denorms(self):
    return self._en_denorm

  def get_en_inf(self):
    return self._en_inf

  def is_packed_float(self):
    return self._gfloat_as_int

  @staticmethod
  def get_gfloat_param_size(calc_type):
    if calc_type == dtypes.float32:
      param_size = cast_to_gfloat32_param_size
    elif calc_type == dtypes.float16:
      param_size = cast_to_gfloat16_param_size
    else:
      raise RuntimeError("Gfloat format calculation type not supported")
    return param_size


def cast_native_to_gfloat(inputs,
                          params,
                          en_nanoo=True,
                          round_mode='RN',
                          sr_bits=23,
                          sr_density='Invalid',
                          sr_norm_offset=0.0,
                          sr_norm_scale=1.0,
                          sr_norm_min=-0.5,
                          sr_norm_max=0.5,
                          sr_prob=1.0,
                          gfloat_format='Invalid',
                          calc_type=dtypes.float32,
                          out_type=dtypes.float32):
  """Quantize input tensor to generic float format
  Args:
    inputs: a tensor with arbitrary shape.
    params: a tensor with Parameters.
    en_nanoo: generate qNan on overflow if Gfloat's Inf is enabled.
    round_mode: Rounding mode for mantissa quantization.
    sr_density: Stochastic rounding noise density
    sr_bits: Number of bits used for stochastic rounding.
    sr_norm_offset: Stochastic rounding noise offset
    sr_norm_scale: Stochastic rounding noise scaling factor
    sr_norm_min: Stochastic rounding noise minimum value
    sr_norm_max: Stochastic rounding noise maximum value
    sr_prob: Bernoulli truncate probability
    gfloat_format: Gfloat format
    calc_type: Native IPU type used for calculation
    out_type: Output storage type
  Returns:
    a tensor of the same shape as the input
  Raises:
  """

  return gen_popfloat_ops.cast_native_to_gfloat(input=inputs,
                                                params=params,
                                                en_nanoo=en_nanoo,
                                                round_mode=round_mode,
                                                sr_density=sr_density,
                                                sr_bits=sr_bits,
                                                sr_norm_offset=sr_norm_offset,
                                                sr_norm_scale=sr_norm_scale,
                                                sr_norm_min=sr_norm_min,
                                                sr_norm_max=sr_norm_max,
                                                sr_prob=sr_prob,
                                                gfloat_format=gfloat_format,
                                                calc_type=calc_type,
                                                out_type=out_type)


def cast_gfloat_to_native(inputs,
                          params,
                          gfloat_format='Invalid',
                          calc_type=dtypes.float32,
                          out_type=dtypes.float32):
  """Unpack a gfloat tensor to FP32 or FP16
  Args:
    inputs: a tensor with arbitrary shape.
    params: cast op parameter tensor.
    gfloat_format: gfloat format type
    calc_type: Native IPU floating point format used for calculation
    out_type: Cast output dtype
  Returns:
    a tensor of the same shape as the input
  """

  outputs = gen_popfloat_ops.cast_gfloat_to_native(input=inputs,
                                                   params=params,
                                                   gfloat_format=gfloat_format,
                                                   out_type=out_type,
                                                   calc_type=calc_type)
  return outputs


def calc_gfloat_params(man=10,
                       exp=5,
                       bias=15,
                       en_denorm=True,
                       en_inf=True,
                       calc_type=dtypes.float32):
  return gen_popfloat_ops.calc_gfloat_params(ops.convert_to_tensor(
      [GenericFloat.get_gfloat_param_size(calc_type)]),
                                             mantissa=man,
                                             exponent=exp,
                                             bias=bias,
                                             en_denorm=en_denorm,
                                             en_inf=en_inf,
                                             calc_type=calc_type)

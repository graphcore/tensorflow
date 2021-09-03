/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_util.h"

#include <string>
#include <utility>

#include "absl/strings/str_replace.h"
#include "tensorflow/core/platform/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {
std::string GetTemplateHloString(const std::string& wu, int n, int m,
                                 int minibatches) {
  const std::string hlo = R"(
  HloModule main

  _pop_op_implicit_binary.add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    bx = f32[$N,$M] broadcast(x), dimensions={}
    by = f32[$N,$M] broadcast(y), dimensions={}
    ROOT add = f32[$N,$M] add(bx, by)
  }

  _pop_op_implicit_binary.mul.1 {
    x = f32[$N,$M] parameter(0)
    y = f32[] parameter(1)
    by = f32[$N,$M] broadcast(y), dimensions={}
    ROOT add = f32[$N,$M] multiply(x, by)
  }

  _pop_op_implicit_binary.mul.2 {
    x = f32[$N,$M] parameter(0)
    y = f32[] parameter(1)
    by = f32[$N,$M] broadcast(y), dimensions={}
    ROOT add = f32[$N,$M] multiply(x, by)
  }

  _pop_op_implicit_binary.mul.3 {
    x = f32[$N,$M] parameter(0)
    y = f32[] parameter(1)
    by = f32[$N,$M] broadcast(y), dimensions={}
    ROOT add = f32[$N,$M] multiply(x, by)
  }

  sum {
    y = f32[] parameter(1)
    x = f32[] parameter(0), control-predecessors={y}
    ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  _pop_op_reduction_square_add {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[] parameter(1)
    mul = f32[$N,$M] multiply(arg0, arg0)
    ROOT r = f32[] reduce(f32[$N,$M] mul, f32[] arg1), dimensions=$REDUCE_DIMS, to_apply=sum
  }

  _pop_op_reduction_square_add.1 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[] parameter(1)
    mul = f32[$N,$M] multiply(arg0, arg0)
    ROOT r = f32[] reduce(f32[$N,$M] mul, f32[] arg1), dimensions=$REDUCE_DIMS, to_apply=sum
  }

  resource_update {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[$N,$M] parameter(2)
    arg3 = f32[$N,$M] parameter(3)

    $WU
  }

  loop {
    after-all = token[] after-all()
    infeed = (f32[$N,$M], token[]) infeed(after-all), infeed_config="\010\001\022\005feed5\"\002\001\001"
    input = f32[$N,$M] get-tuple-element(infeed), index=0

    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[$N,$M] parameter(2)
    arg3 = f32[$N,$M] parameter(3)
    add.1 = f32[$N,$M] add(input, arg0)
    add.2 = f32[$N,$M] add(input, arg1)
    call = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) call(add.1, add.2, arg2, arg3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\", \"partitionOffloadedVariables\":\"THREESTATE_ON\", \"num_batches_to_accumulate\": $B}}}"
    gte0 = f32[$N,$M] get-tuple-element(call), index=0
    gte1 = f32[$N,$M] get-tuple-element(call), index=1
    gte2 = f32[$N,$M] get-tuple-element(call), index=2
    gte3 = f32[$N,$M] get-tuple-element(call), index=3
    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(gte0, gte1, gte2, gte3)
  }

  ENTRY e {
    e.in0 = f32[$N,$M] parameter(0)
    e.in1 = f32[$N,$M] parameter(1)
    e.in2 = f32[$N,$M] parameter(2)
    e.in3 = f32[$N,$M] parameter(3)
    call = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) call(e.in0, e.in1, e.in2, e.in3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
    gte0 = f32[$N,$M] get-tuple-element(call), index=0
    gte1 = f32[$N,$M] get-tuple-element(call), index=1
    gte2 = f32[$N,$M] get-tuple-element(call), index=2
    gte3 = f32[$N,$M] get-tuple-element(call), index=3
    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(gte0, gte1, gte2, gte3)
  }
  )";
  std::string hlo_string =
      tensorflow::str_util::StringReplace(hlo, "$WU", wu, true);
  if (m != 0) {
    hlo_string =
        absl::StrReplaceAll(hlo_string, {{"$N", std::to_string(n)},
                                         {"$M", std::to_string(m)},
                                         {"$B", std::to_string(minibatches)},
                                         {"$REDUCE_DIMS", "{0,1}"}});
  } else {
    hlo_string =
        absl::StrReplaceAll(hlo_string, {{"$N,$M", std::to_string(n)},
                                         {"$B", std::to_string(minibatches)},
                                         {"$REDUCE_DIMS", "{0}"}});
  }
  return hlo_string;
}

}  // namespace

std::string HloPoplarTestUtil::GetLambLikeHloString(int n, int m,
                                                    int minibatches) {
  const std::string wu = R"(
    beta.1 = f32[] constant(0.1)
    beta.2 = f32[] constant(0.01)
    const.55 = f32[] constant(0.9)
    const.2 = f32[] constant(0)
    const.3 = f32[] constant(1)

    all-reduce0 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce1 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    replication-normalise = f32[$N,$M] custom-call(all-reduce0), custom_call_target="ReplicationNormalise"
    custom-call.2 = f32[$N,$M] custom-call(arg2, replication-normalise, const.55), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    constant.54 = f32[] constant(0.001)
    constant.58 = f32[] constant(1)
    subtract.60 = f32[] subtract(constant.58, beta.2)
    sqrt.61 = f32[] sqrt(f32[] subtract.60)
    multiply.62 = f32[] multiply(constant.54, sqrt.61)
    subtract.59 = f32[] subtract(constant.58, beta.1)
    divide.63 = f32[] divide(multiply.62, subtract.59)

    fusion.4 = f32[$N,$M] fusion(custom-call.2, divide.63), kind=kCustom, calls=_pop_op_implicit_binary.mul.1
    multiply.71 = f32[$N,$M] multiply(replication-normalise, replication-normalise)
    custom-call.1 = f32[$N,$M] custom-call(arg3, multiply.71, const.55), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    sqrt.77 = f32[$N,$M] sqrt(custom-call.1)
    const.4 = f32[] constant(1.1)
    fusion.5 = f32[$N,$M] fusion(sqrt.77, const.4), kind=kCustom, calls=_pop_op_implicit_binary.mul.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
    divide.82 = f32[$N,$M] divide(fusion.4, fusion.5)

    wsum = f32[] fusion(all-reduce1, const.2), kind=kCustom, calls=_pop_op_reduction_square_add
    wnorm = f32[] sqrt(wsum)
    %wnormcorrected = f32[] maximum(wnorm, const.3)
    gsum = f32[] fusion(divide.82, const.2), kind=kCustom, calls=_pop_op_reduction_square_add.1
    gnorm = f32[] sqrt(gsum)
    %gnormcorrected = f32[] maximum(gnorm, const.3)
    trustratio = f32[] divide(wnormcorrected, gnormcorrected)
    fusion.7 = f32[$N,$M] fusion(divide.82, trustratio), kind=kCustom, calls=_pop_op_implicit_binary.mul.3

    subtract.83 = f32[$N,$M] subtract(all-reduce1, fusion.7)
    ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(arg0, arg1, subtract.83, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetAdamLikeHloString(int n, int m,
                                                    int minibatches) {
  const std::string wu = R"(
    beta.1 = f32[] constant(0.1)
    beta.2 = f32[] constant(0.01)
    const.55 = f32[] constant(0.9)

    all-reduce0 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce1 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    replication-normalise = f32[$N,$M] custom-call(all-reduce0), custom_call_target="ReplicationNormalise"
    custom-call.2 = f32[$N,$M] custom-call(arg2, replication-normalise, const.55), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    constant.54 = f32[] constant(0.001)
    constant.58 = f32[] constant(1)
    subtract.60 = f32[] subtract(constant.58, beta.2)
    sqrt.61 = f32[] sqrt(f32[] subtract.60)
    multiply.62 = f32[] multiply(constant.54, sqrt.61)
    subtract.59 = f32[] subtract(constant.58, beta.1)
    divide.63 = f32[] divide(multiply.62, subtract.59)

    fusion.4 = f32[$N,$M] fusion(custom-call.2, divide.63), kind=kCustom, calls=_pop_op_implicit_binary.mul.1
    multiply.71 = f32[$N,$M] multiply(replication-normalise, replication-normalise)
    custom-call.1 = f32[$N,$M] custom-call(arg3, multiply.71, const.55), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    sqrt.77 = f32[$N,$M] sqrt(custom-call.1)
    const.4 = f32[] constant(1.1)
    fusion.5 = f32[$N,$M] fusion(sqrt.77, const.4), kind=kCustom, calls=_pop_op_implicit_binary.mul.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
    divide.82 = f32[$N,$M] divide(fusion.4, fusion.5)
    subtract.83 = f32[$N,$M] subtract(all-reduce1, divide.82)
    ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(arg0, arg1, subtract.83, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetMomentumLikeHloString(int n, int m,
                                                        int minibatches) {
  const std::string wu = R"(
  all-reduce = f32[$N,$M] all-reduce(arg0), to_apply=sum
  step = f32[] constant(0.1)
  replication-normalise = f32[$N,$M] custom-call(all-reduce), custom_call_target="ReplicationNormalise"
  custom-call.1 = f32[$N,$M] custom-call(arg2, replication-normalise, step), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
  custom-call.2 = f32[$N,$M] custom-call(arg3, replication-normalise, step), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
  ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(arg0, arg1, custom-call.1, custom-call.2)
)";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetSGDHloString(int n, int m, int minibatches) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    const.1 = f32[] constant(0.1)
    replication-normalise.1 = f32[$N,$M] custom-call(all-reduce.1), custom_call_target="ReplicationNormalise"
    replication-normalise.2 = f32[$N,$M] custom-call(all-reduce.2), custom_call_target="ReplicationNormalise"
    custom-call.1 = f32[$N,$M] custom-call(arg2, replication-normalise.1, const.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    custom-call.2 = f32[$N,$M] custom-call(arg3, replication-normalise.2, const.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, custom-call.1, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetSimpleHloString(int n, int m,
                                                  int minibatches) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum

    add.1 = f32[$N,$M] add(arg2, all-reduce.1)
    add.2 = f32[$N,$M] add(arg3, all-reduce.2)

    rate.1 = f32[] constant(0.1)
    custom-call.1 = f32[$N,$M] custom-call(add.1, add.2, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    custom-call.2 = f32[$N,$M] custom-call(add.2, add.1, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    convert.1 = f16[$N,$M] convert(custom-call.1)
    convert.2 = f32[$N,$M] convert(convert.1)

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, convert.2, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetTwoClustersShareInputHloString(
    int n, int m, int minibatches) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum

    add.1 = f32[$N,$M] add(arg2, all-reduce.1)
    add.2 = f32[$N,$M] add(arg3, all-reduce.1)

    rate.1 = f32[] constant(0.1)
    custom-call.1 = f32[$N,$M] custom-call(add.1, all-reduce.2, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    custom-call.2 = f32[$N,$M] custom-call(add.2, all-reduce.2, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, custom-call.1, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetFullRemoteLoadHloString(int n, int m,
                                                          int minibatches) {
  const std::string wu = R"(
    buffer.1 = f32[$N,$M] custom-call(arg0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}"
    buffer.2 = f32[$N,$M] custom-call(arg1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}"

    add.1 = f32[$N,$M] add(arg2, buffer.1)
    add.2 = f32[$N,$M] add(arg3, buffer.2)

    mul.1 = f32[$N,$M] multiply(add.1, buffer.1)
    mul.2 = f32[$N,$M] multiply(add.2, buffer.2)

    rate.1 = f32[] constant(0.1)
    custom-call.1 = f32[$N,$M] custom-call(mul.1, mul.2, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    custom-call.2 = f32[$N,$M] custom-call(mul.1, mul.2, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    store.1 = f32[$N,$M] custom-call(arg0, custom-call.1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"
    store.2 = f32[$N,$M] custom-call(arg1, custom-call.2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(store.1, store.2, custom-call.1, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetBroadcastHloString(int n, int minibatches) {
  const std::string wu = R"(
    const.1 = f32[] constant(1.0)
    const.2 = f32[] constant(-1.0)
    fusion = f32[$N,$M] fusion(const.1, const.2), kind=kCustom, calls=_pop_op_implicit_binary.add

    add.1 = f32[$N,$M] add(arg2, fusion)
    add.2 = f32[$N,$M] add(arg3, fusion)

    rate.1 = f32[] constant(0.1)
    custom-call.1 = f32[$N,$M] custom-call(add.1, fusion, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"
    custom-call.2 = f32[$N,$M] custom-call(add.2, fusion, rate.1), custom_call_target="ScaledInplaceXbY", backend_config="{\"operation\": \"add\"}"

    convert.1 = f16[$N,$M] convert(custom-call.1)
    convert.2 = f32[$N,$M] convert(convert.1)

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, convert.2, custom-call.2)
  )";
  return GetTemplateHloString(wu, n, 0, minibatches);
}

}  // namespace poplarplugin
}  // namespace xla

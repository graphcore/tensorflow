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

  sum {
    y = f32[] parameter(1)
    x = f32[] parameter(0), control-predecessors={y}
    ROOT add = f32[] add(x, y), backend_config="{\"isInplace\":true}"
  }

  scale_xya.1 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[] parameter(2)
    broadcast = f32[$N,$M] broadcast(arg2), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }

  scale_xya.2 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    arg2 = f32[] parameter(2)
    broadcast = f32[$N,$M] broadcast(arg2), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }

  scale_xa {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[] parameter(1)
    broadcast = f32[$N,$M] broadcast(arg1), dimensions={}
    ROOT mul.1 = f32[$N,$M] multiply(arg0, broadcast)
  }

  inplace_xa {
    arg0 = f32[$N,$M] parameter(0)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    ROOT mul.1 = f32[$N,$M] multiply(arg0, broadcast)
  }

  inplace_xya.1 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
  }
  inplace_xya.2 {
    arg0 = f32[$N,$M] parameter(0)
    arg1 = f32[$N,$M] parameter(1)
    const = f32[] constant(1.1)
    broadcast = f32[$N,$M] broadcast(const), dimensions={}
    mul.1 = f32[$N,$M] multiply(arg0, broadcast)
    ROOT r = f32[$N,$M] add(mul.1, arg1)
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
                                         {"$B", std::to_string(minibatches)}});
  } else {
    hlo_string = absl::StrReplaceAll(
        hlo_string,
        {{"$N,$M", std::to_string(n)}, {"$B", std::to_string(minibatches)}});
  }
  return hlo_string;
}

}  // namespace

std::string HloPoplarTestUtil::GetAdamLikeHloString(int n, int m,
                                                    int minibatches) {
  const std::string wu = R"(
    beta.1 = f32[] constant(0.1)
    beta.2 = f32[] constant(0.01)
    const.55 = f32[] constant(0.9)

    all-reduce0 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce1 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    replication-normalise = f32[$N,$M] custom-call(all-reduce0), custom_call_target="ReplicationNormalise"
    fusion.2 = f32[$N,$M] fusion(arg2, replication-normalise), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"

    constant.54 = f32[] constant(0.001)
    constant.58 = f32[] constant(1)
    subtract.60 = f32[] subtract(constant.58, beta.2)
    sqrt.61 = f32[] sqrt(f32[] subtract.60)
    multiply.62 = f32[] multiply(constant.54, sqrt.61)
    subtract.59 = f32[] subtract(constant.58, beta.1)
    divide.63 = f32[] divide(multiply.62, subtract.59)

    fusion.4 = f32[$N,$M] fusion(fusion.2, divide.63), kind=kCustom, calls=scale_xa
    multiply.71 = f32[$N,$M] multiply(replication-normalise, replication-normalise)
    fusion.1 = f32[$N,$M] fusion(arg3, multiply.71), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
    sqrt.77 = f32[$N,$M] sqrt(fusion.1)
    fusion.5 = f32[$N,$M] fusion(sqrt.77), kind=kCustom, calls=inplace_xa, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
    divide.82 = f32[$N,$M] divide(fusion.4, fusion.5)
    subtract.83 = f32[$N,$M] subtract(all-reduce1, divide.82)
    ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(arg0, arg1, subtract.83, fusion.2)
  )";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetMomentumLikeHloString(int n, int m,
                                                        int minibatches) {
  const std::string wu = R"(
  all-reduce = f32[$N,$M] all-reduce(arg0), to_apply=sum
  replication-normalise = f32[$N,$M] custom-call(all-reduce), custom_call_target="ReplicationNormalise"
  fusion.1 = f32[$N,$M] fusion(arg2, replication-normalise), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
  fusion.2 = f32[$N,$M] fusion(arg3, replication-normalise), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
  ROOT root = (f32[$N,$M], f32[$N,$M], f32[$N,$M], f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
)";
  return GetTemplateHloString(wu, n, m, minibatches);
}

std::string HloPoplarTestUtil::GetSGDHloString(int n, int m, int minibatches) {
  const std::string wu = R"(
    all-reduce.1 = f32[$N,$M] all-reduce(arg0), to_apply=sum
    all-reduce.2 = f32[$N,$M] all-reduce(arg1), to_apply=sum
    replication-normalise.1 = f32[$N,$M] custom-call(all-reduce.1), custom_call_target="ReplicationNormalise"
    replication-normalise.2 = f32[$N,$M] custom-call(all-reduce.2), custom_call_target="ReplicationNormalise"
    fusion.1 = f32[$N,$M] fusion(arg2, replication-normalise.1), kind=kCustom, calls=inplace_xya.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
    fusion.2 = f32[$N,$M] fusion(arg3, replication-normalise.2), kind=kCustom, calls=inplace_xya.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
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
    fusion.1 = f32[$N,$M] fusion(add.1, add.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(add.2, add.1, rate.1), kind=kCustom, calls=scale_xya.2

    convert.1 = f16[$N,$M] convert(fusion.1)
    convert.2 = f32[$N,$M] convert(convert.1)

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, convert.2, fusion.2)
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
    fusion.1 = f32[$N,$M] fusion(add.1, all-reduce.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(add.2, all-reduce.2, rate.1), kind=kCustom, calls=scale_xya.2

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, fusion.1, fusion.2)
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
    fusion.1 = f32[$N,$M] fusion(mul.1, mul.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(mul.1, mul.2, rate.1), kind=kCustom, calls=scale_xya.2

    store.1 = f32[$N,$M] custom-call(arg0, fusion.1), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"
    store.2 = f32[$N,$M] custom-call(arg1, fusion.2), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}"

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(store.1, store.2, fusion.1, fusion.2)
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
    fusion.1 = f32[$N,$M] fusion(add.1, add.2, rate.1), kind=kCustom, calls=scale_xya.1
    fusion.2 = f32[$N,$M] fusion(add.2, add.1, rate.1), kind=kCustom, calls=scale_xya.2

    convert.1 = f16[$N,$M] convert(fusion.1)
    convert.2 = f32[$N,$M] convert(convert.1)

    ROOT r = (f32[$N,$M],f32[$N,$M],f32[$N,$M],f32[$N,$M]) tuple(arg0, arg1, convert.2, fusion.2)
  )";
  return GetTemplateHloString(wu, n, 0, minibatches);
}

}  // namespace poplarplugin
}  // namespace xla

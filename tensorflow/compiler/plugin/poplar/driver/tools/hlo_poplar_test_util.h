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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_UTIL_H_

#include <string>

namespace xla {
namespace poplarplugin {

/*
 Provide minimal training loop and weight update function resembling different
 optimizers.
*/
struct HloPoplarTestUtil {
  static std::string GetAdamLikeHloString(int n, int m, int minibatches = 1);
  static std::string GetMomentumLikeHloString(int n, int m,
                                              int minibatches = 1);
  static std::string GetSGDHloString(int n, int m, int minibatches = 1);
  static std::string GetSimpleHloString(int n, int m, int minibatches = 1);
  static std::string GetTwoClustersShareInputHloString(int n, int m,
                                                       int minibatches = 1);
  static std::string GetFullRemoteLoadHloString(int n, int m,
                                                int minibatches = 1);
  static std::string GetBroadcastHloString(int n, int minibatches = 1);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_UTIL_H_

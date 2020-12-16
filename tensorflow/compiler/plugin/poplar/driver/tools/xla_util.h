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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_XLA_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_XLA_UTIL_H_

namespace xla {
class XlaOp;
class XlaBuilder;

namespace poplarplugin {

// An XLA implementation of the algorithm at
// tensorflow/core/grappler/graph_analyzer/hash_tools.h:CombineHash a.k.a.
// from = from ^ (to + 0x9E3779B9U + (seed << 6) + (seed << 2))
xla::XlaOp CombineHashes(xla::XlaOp from, xla::XlaOp to);

// Hash a seed with the replication index so that each replica has a different
// seed
xla::XlaOp HashSeedWithReplicaIndex(xla::XlaOp seed);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_XLA_UTIL_H_

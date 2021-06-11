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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAKE_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAKE_VISITOR_H_

namespace xla {
namespace poplarplugin {

// Note if using c++17 can do this in 2 lines and should do instead, until that
// day have this make_visitor function instead.

// Mimic of boosts static visitor
template <typename ResultType>
class static_visitor {
 public:
  typedef ResultType result_type;
};

// The variadic template type.
template <typename ReturnType, typename... Lambdas>
struct Visitor;

// A visitor that inherits from a single lambda. This is the terminal element
// of the variadic chain, equivalent to Visitor1 above.
template <typename ReturnType, typename Lambda>
struct Visitor<ReturnType, Lambda> : public static_visitor<ReturnType>,
                                     public Lambda {
  explicit Visitor(Lambda l) : static_visitor<ReturnType>(), Lambda(l) {}

  using Lambda::operator();
};

// A visitor constructed from more than one lambda. It creates a new
// class in the inheritance chain from the first lambda. This is equivalent
// to VisitorN above.
template <typename ReturnType, typename HeadLambda, typename... TailLambdas>
struct Visitor<ReturnType, HeadLambda, TailLambdas...>
    : public Visitor<ReturnType, TailLambdas...>, public HeadLambda {
  Visitor(HeadLambda head, TailLambdas... tail)
      : Visitor<ReturnType, TailLambdas...>(tail...), HeadLambda(head) {}

  using Visitor<ReturnType, TailLambdas...>::operator();
  using HeadLambda::operator();
};

// This is to help template type deduction.
template <typename ReturnType, typename... Lambdas>
Visitor<ReturnType, Lambdas...> make_visitor(Lambdas... lambdas) {
  return Visitor<ReturnType, Lambdas...>(lambdas...);
}

}  // namespace poplarplugin
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAKE_VISITOR_H_

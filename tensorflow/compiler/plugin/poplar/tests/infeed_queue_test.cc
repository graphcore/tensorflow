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

#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"

namespace xla {
namespace poplarplugin {
namespace {

TEST(InfeedQueueTest, PushPop) {
  InfeedQueue q;

  ASSERT_TRUE(q.IsEmpty());
  ASSERT_FALSE(q.IsFull());

  // Push
  tensorflow::Tensor in(1.0f);
  tensorflow::TensorBuffer* inbuf = tensorflow::DMAHelper::buffer(&in);
  inbuf->Ref();
  q.Push(inbuf);
  q.AdvanceWritePosition();

  ASSERT_FALSE(q.IsEmpty());
  ASSERT_FALSE(q.IsFull());

  // Pop
  tensorflow::TensorBuffer* outbuf;
  ASSERT_TRUE(q.TryPop(outbuf));
  ASSERT_EQ(inbuf, outbuf);
  q.AdvanceReadPosition();
  ASSERT_FALSE(q.TryPop(outbuf));
}

TEST(InfeedQueueTest, EndOfQueue) {
  InfeedQueue q;

  q.SignalEndOfQueue();

  tensorflow::TensorBuffer* outbuf;
  ASSERT_FALSE(q.TryPop(outbuf));
  ASSERT_FALSE(q.BlockPop(outbuf));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

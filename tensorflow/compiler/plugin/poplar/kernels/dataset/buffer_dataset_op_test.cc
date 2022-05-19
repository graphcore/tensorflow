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
#include "tensorflow/compiler/plugin/poplar/kernels/dataset/buffer_dataset_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "buffer_dataset";

class BufferDatasetOpTest : public DatasetOpsTestBase {};

class BufferDatasetParams : public DatasetParams {
 public:
  template <typename T>
  BufferDatasetParams(T input_dataset_params, int64_t buffer_size,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        buffer_size_(buffer_size) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {buffer_size_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(BufferDatasetOp::kInputDataset);
    input_names->emplace_back(BufferDatasetOp::kBufferSize);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back(BufferDatasetOp::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(BufferDatasetOp::kOutputShapes, output_shapes_);
    return Status::OK();
  }

  string dataset_type() const override { return BufferDatasetOp::kDatasetType; }

 private:
  const int64_t buffer_size_;
};

// Test case 1: buffer size divides the input size.
BufferDatasetParams BufferSizeDividesSizeParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return BufferDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/5,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: buffer size doesn't divide the input size.
BufferDatasetParams BufferSizeDoesntDividesSizeParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return BufferDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/3,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<BufferDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/BufferSizeDividesSizeParams(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/BufferSizeDoesntDividesSizeParams(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape{1},
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_GET_NEXT_TEST_P(BufferDatasetOpTest, BufferDatasetParams,
                         GetNextTestCases())

TEST_F(BufferDatasetOpTest, DatasetNodeName) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(BufferDatasetOpTest, DatasetTypeString) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(BufferDatasetOp::kDatasetType)));
}

TEST_F(BufferDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(BufferDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<BufferDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/BufferSizeDividesSizeParams(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/BufferSizeDoesntDividesSizeParams(),
           /*expected_cardinality=*/9}};
}

DATASET_CARDINALITY_TEST_P(BufferDatasetOpTest, BufferDatasetParams,
                           CardinalityTestCases())

TEST_F(BufferDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(BufferDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(BufferDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = BufferSizeDividesSizeParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      BufferDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<BufferDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/BufferSizeDividesSizeParams(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/BufferSizeDoesntDividesSizeParams(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape{1},
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(BufferDatasetOpTest, BufferDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

// Test case: buffer size is invalid.
BufferDatasetParams InvalidBufferSizeParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return BufferDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-2,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

TEST_F(BufferDatasetOpTest, InvalidBufferSizeParams) {
  auto dataset_params = InvalidBufferSizeParams();
  EXPECT_EQ(Initialize(dataset_params).code(), error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

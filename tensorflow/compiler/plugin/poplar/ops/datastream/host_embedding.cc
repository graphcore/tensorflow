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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("IpuHostEmbeddingRegister")
    .Input("ref: Ref(T)")
    .Output("output_ref: Ref(T)")
    .Attr("device_ordinal: int = 0")
    .Attr("embedding_id: string")
    .Attr("optimizer: {'SGD', 'SGD+GA'} = 'SGD'")
    .Attr("T: numbertype")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle embedding_shape = c->input(0);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(embedding_shape, 2, &out));

      return shape_inference::UnchangedShape(c);
    });

REGISTER_OP("IpuHostEmbeddingDeregister")
    .Input("ref: Ref(T)")
    .Output("output_ref: Ref(T)")
    .Attr("device_ordinal: int = 0")
    .Attr("embedding_id: string")
    .Attr("T: numbertype")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle embedding_shape = c->input(0);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(embedding_shape, 2, &out));

      return shape_inference::UnchangedShape(c);
    });

REGISTER_OP("IpuDeviceEmbeddingLookup")
    .Input("indices: T")
    .Output("output: dtype")
    .Attr("embedding_id: string")
    .Attr("embedding_shape: shape")
    .Attr("partition_strategy: {'ENCODING', 'TOKEN'} = 'ENCODING'")
    .Attr("dtype: type")
    .Attr("T: {int32}")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape result_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("embedding_shape", &result_shape));

      ShapeHandle indices = c->input(0);
      ShapeHandle out;
      // Require rank 1 indices
      TF_RETURN_IF_ERROR(c->WithRank(indices, 1, &out));

      DimensionHandle dim = c->Dim(indices, 0);
      if (c->ValueKnown(dim)) {
        result_shape.RemoveDim(0);
        result_shape.InsertDim(0, c->Value(dim));
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(result_shape, &out));
        c->set_output(0, out);
      } else {
        return errors::InvalidArgument("Unknown index tensor shape");
      }
      return Status::OK();
    });

REGISTER_OP("IpuDeviceEmbeddingLookupTrainable")
    .Input("dummy: dtype")
    .Input("indices: T")
    .Output("output: dtype")
    .Attr("embedding_id: string")
    .Attr("embedding_shape: shape")
    .Attr("partition_strategy: {'ENCODING', 'TOKEN'} = 'ENCODING'")
    .Attr("dtype: type")
    .Attr("T: {int32}")
    .Attr("optimizer: {'SGD', 'SGD+GA'}")
    .Attr("learning_rate: float")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape result_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("embedding_shape", &result_shape));

      ShapeHandle indices = c->input(1);
      ShapeHandle out;
      // Require rank 1 indices
      TF_RETURN_IF_ERROR(c->WithRank(indices, 1, &out));

      DimensionHandle dim = c->Dim(indices, 0);
      if (c->ValueKnown(dim)) {
        result_shape.RemoveDim(0);
        result_shape.InsertDim(0, c->Value(dim));
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(result_shape, &out));
        c->set_output(0, out);
      } else {
        return errors::InvalidArgument("Unknown index tensor shape");
      }
      return Status::OK();
    });

REGISTER_OP("IpuDeviceEmbeddingUpdateAdd")
    .Input("ins: T")
    .Input("grads: T")
    .Input("indices: Tindices")
    .Attr("embedding_id: string")
    .Attr("embedding_shape: shape")
    .Attr("partition_strategy: {'ENCODING', 'TOKEN'} = 'ENCODING'")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32}")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle ins = c->input(0);
      ShapeHandle grads = c->input(1);
      ShapeHandle out;
      // Require rank 2 grads
      TF_RETURN_IF_ERROR(c->WithRank(grads, 2, &out));
      TF_RETURN_IF_ERROR(c->WithRank(ins, 2, &out));

      PartialTensorShape embedding_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("embedding_shape", &embedding_shape));

      // Require rank 1 indices
      ShapeHandle indices = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(indices, 1, &out));

      DimensionHandle indices_dim_0 = c->Dim(indices, 0);
      DimensionHandle grads_dim_0 = c->Dim(grads, 0);
      DimensionHandle grads_dim_1 = c->Dim(grads, 1);
      DimensionHandle ins_dim_0 = c->Dim(ins, 0);
      DimensionHandle ins_dim_1 = c->Dim(ins, 1);

      if (!c->ValueKnown(indices_dim_0)) {
        return errors::InvalidArgument(
            "Dimension 0 of the indices does not have a known value.");
      }

      if (!c->ValueKnown(ins_dim_0)) {
        return errors::InvalidArgument(
            "Dimension 0 of the ins does not have a known value.");
      }

      if (!c->ValueKnown(grads_dim_0)) {
        return errors::InvalidArgument(
            "Dimension 0 of the updates does not have a known value.");
      }

      if (!embedding_shape.IsFullyDefined()) {
        return errors::InvalidArgument(
            "The embedding shape is not fully defined.");
      }

      if (c->Value(indices_dim_0) != c->Value(grads_dim_0)) {
        return errors::InvalidArgument(
            "Dimension 0 of the indices and updates do not match ",
            c->Value(indices_dim_0), " != ", c->Value(grads_dim_0), ".");
      }

      if (!c->ValueKnown(ins_dim_1)) {
        return errors::InvalidArgument(
            "Dimension 1 of the ins does not have a known value.");
      }

      if (!c->ValueKnown(grads_dim_1)) {
        return errors::InvalidArgument(
            "Dimension 1 of the updates does not have a known value.");
      }

      if (c->Value(grads_dim_1) != embedding_shape.dim_size(1)) {
        return errors::InvalidArgument(
            "Dimension 1 of the updates and embedding_shape do not match ",
            c->Value(grads_dim_1), " != ", embedding_shape.dim_size(1), ".");
      }

      if (c->Value(ins_dim_0) != c->Value(grads_dim_0)) {
        return errors::InvalidArgument(
            "Dimension 0 of the ins and updates do not match ",
            c->Value(ins_dim_0), " != ", c->Value(grads_dim_0), ".");
      }

      if (c->Value(ins_dim_1) != c->Value(grads_dim_1)) {
        return errors::InvalidArgument(
            "Dimension 1 of the ins and updates do not match ",
            c->Value(ins_dim_1), " != ", c->Value(grads_dim_0), ".");
      }

      return shape_inference::NoOutputs(c);
    });

REGISTER_OP("IpuDeviceEmbeddingNotify")
    .Attr("device_ordinal: int = 0")
    .Attr("embedding_id: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/ops/common_shape_fns.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("PopDatastreamInfeedDequeue")
    .Output("outputs: output_types")
    .Attr("feed_id: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("prefetch_depth: int = 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
A placeholder op for multiple values that will be fed into the computation
simultaneously as an XLA tuple.

outputs: A list of tensors that will be provided using the infeed mechanism.
feed_id: The id of the iterator used for this dequeue.
output_types: The element types of each element in `outputs`.
output_shapes: The shapes of each tensor in `outputs`.
prefetch_depth: the number of elements poplar will prefetch.
  The depth of the poplar datastream buffer size which may be prefetched
  before being read by the device. By default the prefetch_depth size is
  1, so prefetches a single entry after it has been read to refill the
  buffer. Increasing the size of the prefetch_depth allows for prefetching
  of multiple entries, increasing the probability there will be a valid
  entry in the buffer for the device to read before falling back to
  synchronously fetching the next entry.
)doc");

REGISTER_OP("IPUCreateDatasetIterator")
    .Input("input_dataset: variant")
    .Attr("device_ordinal: int = 0")
    .Attr("feed_id: string")
    .Attr("prefetch_depth: int = 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("IPUDeleteDatasetIterator")
    .Attr("device_ordinal: int = 0")
    .Attr("feed_id: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("PopDatastreamOutfeedEnqueue")
    .Input("inputs: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("outfeed_mode: string='all'")
    .Attr("feed_id: string")
    .Attr("prefetch_depth: int = 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
An op which emits multiple Tensor values from an XLA computation.

inputs: A list of tensors that will be inserted into the outfeed queue as an
XLA tuple.
output_types: The element types of each element in `outputs`.
outfeed_mode: 'all' or 'get_last', default is 'all'. In 'all'-mode all outfed
  values are enqueued for reading on the host. In 'get_last'-mode a single
  value is queued to be passed to the host.
)doc");

REGISTER_OP("PopDatastreamOutfeedDequeue")
    .Output("outputs: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("device_ordinal: int = 0")
    .Attr("feed_id: string")
    .Attr("outfeed_mode: string='all'")
    .Attr("prefetch_depth: int = 1")
    .Attr("warn_when_unconnected: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Retrieve multiple values that will be emitted by the computation as an XLABLA
tuple.  This operations will block indefinitely until data is available.
Output `i` corresponds to XLA tuple element `i`.

outputs: A list of tensors that will be read from the outfeed.
output_types: The element types of each element in `outputs`.
output_shapes: The output_shapes of each tensor in `outputs`. If the first
  dimension is None, then the dequeue operation will output all outfed
  elements available.
device_ordinal: The IPU device to use.
)doc");

REGISTER_OP("IPUDeleteOutfeed")
    .Attr("device_ordinal: int = 0")
    .Attr("feed_id: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow

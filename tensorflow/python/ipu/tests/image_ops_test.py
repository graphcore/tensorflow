# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import numpy as np
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ImageOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testNormaliseImage(self):
    NUM_IMAGES = 3

    def make_graph(offsets, scales, scale=1, im_type=None, im_shape=None):
      im_shape = im_shape or [2, 2, 2, 3]
      dataset = tu.create_single_increasing_dataset(NUM_IMAGES,
                                                    shape=im_shape,
                                                    dtype=np.float32)

      infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def body(image):
        if im_type:
          image = math_ops.cast(image, im_type)
        normalised = ipu.ops.image_ops.normalise_image(image, offsets, scales,
                                                       scale)
        enqueue = outfeed.enqueue(normalised)
        return enqueue

      def my_net():
        return ipu.loops.repeat(NUM_IMAGES, body, [], infeed)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        return ipu.ipu_compiler.compile(my_net), infeed, outfeed

    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def test_case(offsets,
                    scales,
                    scale=1,
                    im_type=None,
                    im_shape=None,
                    tensor_scales_offsets=False):
        im_shape = im_shape or [2, 2, 2, 3]

        offsets_t = offsets
        scales_t = scales
        if tensor_scales_offsets:
          offsets_t = constant_op.constant(offsets)
          scales_t = constant_op.constant(scales)

        run, inf, outf = make_graph(offsets_t, scales_t, scale, im_type,
                                    im_shape)
        sess.run(inf.initializer)
        sess.run(run)
        results = sess.run(outf.dequeue())

        # Calculate expected results:
        # Make n images which have linearly increasing blanket values.
        expected = (np.ones([1] + im_shape).T * np.arange(NUM_IMAGES)).T

        # Cast and normalize (elementwise, then broadcasted scales and offsets).
        expected = ((expected.astype(im_type) * scale) - offsets) * scales

        # Pad to 4 channels.
        padding = np.zeros([NUM_IMAGES] + im_shape[:-1] + [4 - im_shape[-1]])
        expected = np.c_[expected, padding]

        self.assertAllClose(results, expected)

      # Simple usage in float32.
      test_case(np.array([1, 2, 3], np.float32),
                np.array([4, 5, 6], np.float32),
                scale=2)

      # Strange but valid shape.
      test_case(np.array([1, 2, 3], np.float32),
                np.array([4, 5, 6], np.float32),
                im_shape=[2, 1, 2, 9, 3])

      # Only 2 channels.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "The image has 2 channels, expected 3."):
        test_case(np.array([1, 2, 3], np.float32),
                  np.array([4, 5, 6], np.float32),
                  im_shape=[2, 2])

      # Bad shapes for scales/offsets.
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "must be the same size as the number of image channels 3,"
          " but was"):
        test_case(np.array([1, 2], np.float32), np.array([4, 5, 6],
                                                         np.float32))
        test_case(np.array([1, 2, 3], np.float32),
                  np.array([4, 5, 6, 7], np.float32))

      # Precise and negative values.
      test_case(np.array([3.82, -1.9999, 6000], np.float32),
                np.array([-1, 1.5, 6.3333], np.float32),
                scale=-3.283)

      # float16.
      test_case(np.array([1, 2, 3], np.float16),
                np.array([4, 5, 6], np.float16), 2, np.float16)

      # uint8.
      test_case(np.array([1, 2, 3], np.float16),
                np.array([4, 5, 6], np.float16), 2, np.uint8)

      # Differing types are automatically handled.
      # float16 scales/offsets --> float32 image.
      test_case(np.array([1, 2, 3], np.float16),
                np.array([4, 5, 6], np.float16), 2, np.float32)

      # float32 scales/offsets --> uint8 image.
      test_case(np.array([1, 2, 3], np.float32),
                np.array([4, 5, 6], np.float32), 2, np.uint8)

      # They're also handled for tensor scales and offsets.
      test_case(np.array([1, 2, 3], np.float32),
                np.array([4, 5, 6], np.float32),
                2,
                np.uint8,
                tensor_scales_offsets=True)


if __name__ == "__main__":
  googletest.main()

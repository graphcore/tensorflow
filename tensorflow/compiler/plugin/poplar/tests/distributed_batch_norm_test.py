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
# ==============================================================================

from absl.testing import parameterized
from tensorflow.python.ipu.config import IPUConfig
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
import test_utils as tu


# pylint: disable=abstract-method
class DistributedBatchNormTest(xla_test.XLATestCase, parameterized.TestCase):
  def _reference_training(self, acts, scale, offset, epsilon, replicas,
                          group_size):
    norm_batch_size = acts.shape[0] * group_size // replicas
    num_groups = acts.shape[0] // norm_batch_size
    normalized_vals = []
    mean_vals = []
    var_vals = []
    for i in range(num_groups):
      x = acts[i * norm_batch_size:(i + 1) * norm_batch_size]
      x_square = x * x
      x_square_sum = np.sum(x_square, (0, 1, 2))
      x_sum = np.sum(x, axis=(0, 1, 2))
      element_count = np.size(x) / int(np.shape(x)[-1])
      mean = x_sum / element_count
      var = x_square_sum / element_count - mean * mean
      normalized = (x - mean) / np.sqrt(var + epsilon)
      normalized = normalized * scale + offset

      # When in training mode, fused_batchnorm applies an implicit Bessel's
      # correction.
      factor = element_count / max(element_count - 1, 1)
      corrected_var = var * factor

      normalized_vals.append(np.expand_dims(normalized, 0))
      mean_vals.append(np.expand_dims(mean, 0))
      var_vals.append(np.expand_dims(corrected_var, 0))
    return np.concatenate(normalized_vals), np.concatenate(
        mean_vals), np.concatenate(var_vals)

  def _reference_grad(self, grads, acts, scale, means, var_vals, epsilon,
                      replicas, batch_size, group_size):
    norm_batch_size = batch_size * group_size
    num_groups = acts.shape[0] // norm_batch_size

    grad_acts = []
    for i in range(num_groups):
      grad = grads[i * norm_batch_size:(i + 1) * norm_batch_size]
      x = acts[i * norm_batch_size:(i + 1) * norm_batch_size]
      mean = means[i]
      var = var_vals[i]
      inv_std_dev = 1.0 / np.sqrt(var + epsilon)
      # grad_x =
      #   1/N * scale * rsqrt(var + epsilon) * (N * grad - sum(grad) -
      #   (x - mean) * sum(grad * (x - mean)) / (var + epsilon))
      grad_x = scale * (grad - np.mean(grad, axis=(0, 1, 2)) -
                        (x - mean) * np.mean(grad *
                                             (x - mean), axis=(0, 1, 2)) /
                        (var + epsilon)) * inv_std_dev

      grad_acts.append(np.expand_dims(grad_x, 0))

    grad_scales = []
    grad_offsets = []
    for i in range(replicas):
      group_idx = i // group_size
      grad = grads[i * batch_size:(i + 1) * batch_size]
      x = acts[i * batch_size:(i + 1) * batch_size]
      mean = means[group_idx]
      var = var_vals[group_idx]
      inv_std_dev = 1.0 / np.sqrt(var + epsilon)
      # grad_scale =
      #   sum(grad * (x - mean)) * rsqrt(var + epsilon)
      # grad_offset = sum(grad)

      grad_scale = np.sum(grad * (x - mean) * inv_std_dev, axis=(0, 1, 2))

      grad_offset = np.sum(grad, axis=(0, 1, 2))

      grad_scales.append(np.expand_dims(grad_scale, 0))
      grad_offsets.append(np.expand_dims(grad_offset, 0))

    return np.concatenate(grad_acts), np.concatenate(
        grad_scales), np.concatenate(grad_offsets)

  def _configure(self, replicas, group_size):
    config = IPUConfig()
    config.auto_select_ipus = replicas
    tu.add_hw_ci_connection_options(config)
    config.norms.experimental.distributed_batch_norm_replica_group_size = \
        group_size
    config.configure_ipu_system()

  @parameterized.parameters([np.float32, np.float16])
  @tu.test_uses_ipus(num_ipus=8)
  def testBatchNormalize(self, dtype):
    np.random.seed(32)
    rtol = 0.2 if dtype == np.float16 else 0.1
    atol = 7e-2 if dtype == np.float16 else 1e-3

    replicas = 8
    group_size = 2
    batch_size = 4

    channel = 3
    spatial_dims = [8, 8]
    acts_shape = [replicas * batch_size] + spatial_dims + [channel]
    scale_shape = [channel]
    acts_val = np.random.random_sample(acts_shape).astype(dtype)
    scale_val = np.random.random_sample(scale_shape).astype(dtype)
    offset_val = np.random.random_sample(scale_shape).astype(dtype)
    epsilon = 0.001

    self._configure(replicas, group_size)

    normalised_refs, mean_refs, var_refs = self._reference_training(
        acts_val, scale_val, offset_val, epsilon, replicas, group_size)

    fwd_outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def fwd_fn(acts, scale, offset):
      # Slice out the input based on the replica index.
      rep_id = ipu.replication_ops.replication_index()
      acts = array_ops.slice(acts,
                             begin=[batch_size * rep_id, 0, 0, 0],
                             size=[batch_size] + spatial_dims + [channel])
      vals = nn.fused_batch_norm(acts,
                                 scale,
                                 offset,
                                 mean=None,
                                 variance=None,
                                 epsilon=epsilon,
                                 is_training=True)
      return fwd_outfeed.enqueue(vals)

    with self.session() as sess, self.test_scope():
      acts = array_ops.placeholder(dtype, shape=acts_val.shape, name="acts")
      scale = array_ops.placeholder(dtype, shape=scale_shape, name="scale")
      offset = array_ops.placeholder(dtype, shape=scale_shape, name="offset")

      compiled = ipu.ipu_compiler.compile(fwd_fn, [acts, scale, offset])

      sess.run(compiled, {
          acts: acts_val,
          scale: scale_val,
          offset: offset_val
      })
      normalised_acts, mean_vals, var_vals = sess.run(fwd_outfeed.dequeue())
      # There is only a single iteration.
      normalised_acts = normalised_acts[0]
      mean_vals = mean_vals[0]
      var_vals = var_vals[0]

      for i in range(0, replicas, group_size):
        group_idx = i // group_size
        group_normalised_acts = normalised_acts[i:i + group_size]
        group_normalised_acts = np.reshape(group_normalised_acts,
                                           [-1] + spatial_dims + [channel])
        self.assertAllClose(group_normalised_acts,
                            normalised_refs[group_idx],
                            rtol=rtol,
                            atol=atol)
        for j in range(group_size):
          self.assertAllClose(mean_vals[i + j],
                              mean_refs[group_idx],
                              rtol=rtol,
                              atol=atol)
          self.assertAllClose(var_vals[i + j],
                              var_refs[group_idx],
                              rtol=rtol,
                              atol=atol)

    # Run the gradient.
    grad_val = np.random.random_sample(acts_shape).astype(dtype)

    grad_vals_refs, grad_scales_refs, grad_offsets_refs = self._reference_grad(
        grad_val, acts_val, scale_val, mean_refs, var_refs, epsilon, replicas,
        batch_size, group_size)

    grad_outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def grad_fn(grad, acts, scale, mean, var):
      # Slice out the values based on replica index.
      rep_id = ipu.replication_ops.replication_index()
      grad = array_ops.slice(grad,
                             begin=[batch_size * rep_id, 0, 0, 0],
                             size=[batch_size] + spatial_dims + [channel])
      acts = array_ops.slice(acts,
                             begin=[batch_size * rep_id, 0, 0, 0],
                             size=[batch_size] + spatial_dims + [channel])
      group_idx = rep_id // group_size
      mean = array_ops.slice(mean, begin=[group_idx, 0], size=[1, channel])
      mean = array_ops.squeeze(mean, 0)
      var = array_ops.slice(var, begin=[group_idx, 0], size=[1, channel])
      var = array_ops.squeeze(var, 0)

      grad_acts, grad_scale, grad_offset, _, _ = \
          gen_nn_ops.fused_batch_norm_grad_v2(
              grad,
              acts,
              scale,
              mean,
              var,
              data_format="NHWC",
              is_training=True)

      return grad_outfeed.enqueue([grad_acts, grad_scale, grad_offset])

    with self.session() as sess, self.test_scope():
      grad = array_ops.placeholder(dtype, shape=acts_shape, name="grad")
      acts = array_ops.placeholder(dtype, shape=acts_shape, name="acts")
      scale = array_ops.placeholder(dtype, shape=scale_shape, name="scale")
      mean = array_ops.placeholder(dtype, shape=mean_refs.shape, name="mean")
      var = array_ops.placeholder(dtype, shape=var_refs.shape, name="var")

      compiled = ipu.ipu_compiler.compile(grad_fn,
                                          [grad, acts, scale, mean, var])

      sess.run(
          compiled, {
              grad: grad_val,
              acts: acts_val,
              scale: scale_val,
              mean: mean_refs,
              var: var_refs,
          })
      grad_acts, grad_scales, grad_offsets = sess.run(grad_outfeed.dequeue())
      # There is only a single iteration.
      grad_acts = grad_acts[0]
      grad_scales = grad_scales[0]
      grad_offsets = grad_offsets[0]

      for i in range(0, replicas, group_size):
        group_idx = i // group_size
        group_grad_acts = grad_acts[i:i + group_size]
        group_grad_acts = np.reshape(group_grad_acts,
                                     [-1] + spatial_dims + [channel])
        self.assertAllClose(group_grad_acts,
                            grad_vals_refs[group_idx],
                            rtol=rtol,
                            atol=atol)

      self.assertAllClose(grad_scales, grad_scales_refs, rtol=rtol, atol=atol)
      self.assertAllClose(grad_offsets,
                          grad_offsets_refs,
                          rtol=rtol,
                          atol=atol)


if __name__ == "__main__":
  test.main()

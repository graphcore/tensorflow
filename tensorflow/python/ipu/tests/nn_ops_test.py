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

from functools import partial
import numpy as np

from tensorflow.python.client import session as se
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.ops import nn_ops
from tensorflow.python.ipu.ops.nn_ops import _compute_sampled_logits


class ComputeSampledLogitsTest(test_util.TensorFlowTestCase):
  def setUp(self):
    self._eps = 1e-3

  def _GenerateTestData(self, num_classes, dim, batch_size, num_true, labels,
                        sampled, subtract_log_q):
    """Randomly generates input/output data for a single test case.

    This function returns numpy constants for use in a test case.

    Args:
      num_classes: An int. The number of embedding classes in the test case.
      dim: An int. The dimension of the embedding.
      batch_size: An int. The batch size.
      num_true: An int. The number of target classes per training example.
      labels: A list of batch_size * num_true ints. The target classes.
      sampled: A list of indices in [0, num_classes).
      subtract_log_q: A bool corresponding to the parameter in
          _compute_sampled_logits().

    Returns:
      weights: Embedding weights to use as test input. It is a numpy array
          of shape [num_classes, dim]
      biases: Embedding biases to use as test input. It is a numpy array
          of shape [num_classes].
      hidden_acts: Forward activations of the network to use as test input.
          It is a numpy array of shape [batch_size, dim].
      sampled_vals: A tuple based on `sampled` to use as test input in the
          format returned by a *_candidate_sampler function.
      exp_logits: The output logits expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
      exp_labels: The output labels expected from _compute_sampled_logits().
          It is a numpy array of shape [batch_size, num_true + len(sampled)].
    """
    weights = np.random.randn(num_classes, dim).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    hidden_acts = np.random.randn(batch_size, dim).astype(np.float32)

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_vals = (sampled, true_exp, sampled_exp)

    sampled_w, sampled_b = weights[sampled], biases[sampled]
    true_w, true_b = weights[labels], biases[labels]

    true_logits = np.sum(hidden_acts.reshape(
        (batch_size, 1, dim)) * true_w.reshape((batch_size, num_true, dim)),
                         axis=2)
    true_b = true_b.reshape((batch_size, num_true))
    true_logits += true_b
    sampled_logits = np.dot(hidden_acts, sampled_w.T) + sampled_b

    if subtract_log_q:
      true_logits -= np.log(true_exp)
      sampled_logits -= np.log(sampled_exp[np.newaxis, :])

    exp_logits = np.concatenate([true_logits, sampled_logits], axis=1)
    exp_labels = np.hstack(
        (np.ones_like(true_logits) / num_true, np.zeros_like(sampled_logits)))

    return weights, biases, hidden_acts, sampled_vals, exp_logits, exp_labels

  def _SetSeeds(self, seed):
    np.random.seed(seed)
    ipu.utils.reset_ipu_seed(seed)
    random_seed.set_random_seed(seed)

  @test_util.deprecated_graph_mode_only
  def testShapes(self):
    self._SetSeeds(42)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(low=0,
                                 high=num_classes,
                                 size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(num_classes=num_classes,
                                            dim=10,
                                            batch_size=batch_size,
                                            num_true=num_true,
                                            labels=labels,
                                            sampled=[1, 0, 2, 3],
                                            subtract_log_q=False)

      def model(weights, biases, labels, hidden_acts, num_true, sampled_vals):
        return _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(labels,
                                        dtype=dtypes.int64,
                                        shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            name="sampled_logits_basic_num_true_%d" % num_true)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(
            partial(model, weights, biases, labels, hidden_acts, num_true,
                    sampled_vals))
      with se.Session() as sess:
        got_logits, got_labels = sess.run(r)
      self.assertEqual(exp_logits.shape, got_logits.shape, self._eps)
      self.assertEqual(exp_labels.shape, got_labels.shape, self._eps)

  @test_util.deprecated_graph_mode_only
  def testBasic(self):
    """Without accidental hit removal or subtract_log_q."""
    self._SetSeeds(42)
    num_classes = 5
    batch_size = 3

    for num_true in range(1, 5):
      labels = np.random.randint(low=0,
                                 high=num_classes,
                                 size=batch_size * num_true)
      (weights, biases, hidden_acts, sampled_vals, exp_logits,
       exp_labels) = self._GenerateTestData(num_classes=num_classes,
                                            dim=10,
                                            batch_size=batch_size,
                                            num_true=num_true,
                                            labels=labels,
                                            sampled=[1, 0, 2, 3],
                                            subtract_log_q=False)

      def model(weights, biases, labels, hidden_acts, num_true, sampled_vals):
        return _compute_sampled_logits(
            weights=constant_op.constant(weights),
            biases=constant_op.constant(biases),
            labels=constant_op.constant(labels,
                                        dtype=dtypes.int64,
                                        shape=(batch_size, num_true)),
            inputs=constant_op.constant(hidden_acts),
            num_sampled=4,
            num_classes=num_classes,
            num_true=num_true,
            sampled_values=sampled_vals,
            subtract_log_q=False,
            name="sampled_logits_basic_num_true_%d" % num_true)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(
            partial(model, weights, biases, labels, hidden_acts, num_true,
                    sampled_vals))
      with se.Session() as sess:
        got_logits, got_labels = sess.run(r)
      self.assertAllClose(exp_logits, got_logits, self._eps)
      self.assertAllClose(exp_labels, got_labels, self._eps)

  @test_util.deprecated_graph_mode_only
  def testSampledSoftmaxLoss(self):
    def _SoftmaxCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      stable_exp_logits = np.exp(logits -
                                 np.amax(logits, axis=1, keepdims=True))
      pred = stable_exp_logits / np.sum(stable_exp_logits, 1, keepdims=True)
      return -np.sum(targets * np.log(pred + 1.0e-20), axis=1)

    self._SetSeeds(42)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(num_classes=num_classes,
                                          dim=10,
                                          batch_size=batch_size,
                                          num_true=1,
                                          labels=labels,
                                          sampled=[1, 0, 2, 3],
                                          subtract_log_q=True)
    exp_sampled_softmax_loss = _SoftmaxCrossEntropyWithLogits(
        exp_logits, exp_labels)

    def model(weights, biases, labels, hidden_acts, sampled_vals):
      return nn_ops.sampled_softmax_loss(
          weights=constant_op.constant(weights),
          biases=constant_op.constant(biases),
          labels=constant_op.constant(labels, shape=(batch_size, 1)),
          inputs=constant_op.constant(hidden_acts),
          num_sampled=4,
          num_classes=num_classes,
          num_true=1,
          sampled_values=sampled_vals)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(
          partial(model, weights, biases, labels, hidden_acts, sampled_vals))
    with se.Session() as sess:
      got_sampled_softmax_loss = sess.run(r)[0]
    self.assertAllClose(exp_sampled_softmax_loss, got_sampled_softmax_loss,
                        1e-4)

  @test_util.deprecated_graph_mode_only
  def testNCELoss(self):
    def _SigmoidCrossEntropyWithLogits(logits, targets):
      # logits, targets: float arrays of the same shape.
      assert logits.shape == targets.shape
      pred = 1. / (1. + np.exp(-logits))
      eps = 0.0001
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)
      return -targets * np.log(pred) - (1. - targets) * np.log(1. - pred)

    self._SetSeeds(42)
    num_classes = 5
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals, exp_logits,
     exp_labels) = self._GenerateTestData(num_classes=num_classes,
                                          dim=10,
                                          batch_size=batch_size,
                                          num_true=1,
                                          labels=labels,
                                          sampled=[1, 0, 2, 3],
                                          subtract_log_q=True)
    exp_nce_loss = np.sum(
        _SigmoidCrossEntropyWithLogits(exp_logits, exp_labels), 1)

    def model(weights, biases, labels, hidden_acts, sampled_vals):
      return nn_ops.nce_loss(weights=constant_op.constant(weights),
                             biases=constant_op.constant(biases),
                             labels=constant_op.constant(labels,
                                                         shape=(batch_size,
                                                                1)),
                             inputs=constant_op.constant(hidden_acts),
                             num_sampled=4,
                             num_classes=num_classes,
                             num_true=1,
                             sampled_values=sampled_vals)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(
          partial(model, weights, biases, labels, hidden_acts, sampled_vals))
    with se.Session() as sess:
      got_nce_loss = sess.run(r)[0]
    self.assertAllClose(exp_nce_loss, got_nce_loss, 1e-4)


class PopnnGeluTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testGelu(self):
    def test_approx_gelu(x):
      return 0.5 * x * (
          1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def ipu_gelu(x):
      x = ipu.ops.nn_ops.gelu(x)
      return [x]

    for test_type in [[np.float16, 1e-2], [np.float32, 1e-7]]:
      with ops.device('cpu'):
        input_data = array_ops.placeholder(test_type[0], shape=[10, 20])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(ipu_gelu, inputs=[input_data])

      with se.Session() as sess:
        in_data = np.random.rand(10, 20)
        ipu_result = sess.run(r, {input_data: in_data})

      self.assertAllClose(ipu_result, [test_approx_gelu(np.array(in_data))],
                          rtol=test_type[1])

  @test_util.deprecated_graph_mode_only
  def testGeluGrad(self):
    a_size = 5
    b_size = 6
    ab_size = 10
    mat_values = [
        -5.0, -1.2, -1.0, -0.5, -0.2, -0.15, -0.1, 0.0, 0.1, 0.15, 0.2, 0.5,
        1.0, 1.2, 5.0
    ]

    def test_sech(x):
      return 2.0 / (np.exp(x) + np.exp(-x))

    def test_approx_derivative_gelu(x):
      return 0.5 * np.tanh(0.0356774*np.power(x, 2) + 0.797885*x) + \
      (0.0535161*np.power(x, 3) + 0.398942*x)* \
      np.power(test_sech(0.0356774*np.power(x, 3) + 0.797885*x), 2) + 0.5

    def ipu_gelu_back(a, b):
      w = math_ops.matmul(a, b)
      gelu_output = ipu.ops.nn_ops.gelu(w)
      cost = gelu_output
      opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)
      gradients = opt.compute_gradients(cost, w)

      return [gelu_output, gradients, w]

    for mat_value in mat_values:
      for test_type in [[np.float16, 1e-2], [np.float32, 1e-2]]:
        with ops.device('cpu'):
          a = array_ops.placeholder(test_type[0], shape=[a_size, ab_size])
          b = array_ops.placeholder(test_type[0], shape=[ab_size, b_size])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r = ipu.ipu_compiler.compile(ipu_gelu_back, inputs=[a, b])

        with se.Session() as sess:
          in_a = np.full((a_size, ab_size), mat_value)
          in_b = np.full((ab_size, b_size), mat_value)
          res = sess.run(r, {a: in_a, b: in_b})

          gradients_res_values = res[1][0][1]
          gradients_res_grads = res[1][0][0]
          variable_values = res[2]

          self.assertAllClose(variable_values,
                              gradients_res_values,
                              rtol=test_type[1])
          self.assertEqual(gradients_res_grads.shape, (a_size, b_size))
          self.assertAllClose(test_approx_derivative_gelu(mat_value *
                                                          mat_value * ab_size),
                              gradients_res_grads[0][0],
                              rtol=test_type[1])


if __name__ == "__main__":
  googletest.main()

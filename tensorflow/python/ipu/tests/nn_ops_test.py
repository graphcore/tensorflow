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
import absl.testing
from absl.testing import parameterized

from tensorflow.python.client import session as se
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ctc_ops
from tensorflow.python.ops import nn_ops as tf_nn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.ops import nn_ops
from tensorflow.python.ipu.ops.nn_ops import _compute_sampled_logits

from tensorflow.nn import ctc_beam_search_decoder as tf_ctc_beam_search
from tensorflow import sparse


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
      pred = np.minimum(np.maximum(pred, eps), 1 - eps)  # pylint: disable=E1111
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


class Gelu(object):
  op = ipu.ops.nn_ops.gelu

  @staticmethod
  def approx_activation(x):
    return 0.5 * x * (
        1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

  @staticmethod
  def approx_derivative(x):
    test_sech = lambda x: (2.0 / (np.exp(x) + np.exp(-x)))
    return 0.5 * np.tanh(0.0356774*np.power(x, 2) + 0.797885*x) + \
    (0.0535161*np.power(x, 3) + 0.398942*x)* \
    np.power(test_sech(0.0356774*np.power(x, 3) + 0.797885*x), 2) + 0.5


class HardSigmoid(object):
  op = ipu.ops.nn_ops.hard_sigmoid

  @staticmethod
  def approx_activation(x):
    result = 0.2 * x + 0.5
    np.clip(result, 0.0, 1.0)
    return result

  @staticmethod
  def approx_derivative(x):
    return 0.2 * (abs(x) <= 2.5)


class Swish(object):
  op = ipu.ops.nn_ops.swish

  @staticmethod
  def approx_activation(x):
    return x / (1 + np.exp(-x))

  @staticmethod
  def approx_derivative(x):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    x_sigmoid = sigmoid(x)
    return x_sigmoid * (1 + x * (1 - x_sigmoid))


NonLinearities = ("HardSigmoid", HardSigmoid), ("Gelu", Gelu), ("Swish", Swish)


@absl.testing.parameterized.named_parameters(*NonLinearities)
class NonLinearityTest(test_util.TensorFlowTestCase,
                       absl.testing.parameterized.TestCase):
  @test_util.deprecated_graph_mode_only
  def testActivation(self, NonLinearity):
    for test_type in [[np.float16, 1e-2], [np.float32, 1e-7]]:
      with ops.device('cpu'):
        input_data = array_ops.placeholder(test_type[0], shape=[10, 20])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(lambda x: [NonLinearity.op(x)],
                                     inputs=[input_data])

      with se.Session() as sess:
        in_data = np.random.rand(10, 20)
        ipu_result = sess.run(r, {input_data: in_data})

      self.assertAllClose(ipu_result,
                          [NonLinearity.approx_activation(np.array(in_data))],
                          rtol=test_type[1])

  @test_util.deprecated_graph_mode_only
  def testGradient(self, NonLinearity):
    a_size = 5
    b_size = 6
    ab_size = 10
    mat_values = [
        -5.0, -1.2, -1.0, -0.5, -0.2, -0.15, -0.1, 0.0, 0.1, 0.15, 0.2, 0.5,
        1.0, 1.2, 5.0
    ]

    def ipu_back(a, b):
      w = math_ops.matmul(a, b)
      output = NonLinearity.op(w)
      cost = output
      opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)
      gradients = opt.compute_gradients(cost, w)

      return [output, gradients, w]

    for mat_value in mat_values:
      for test_type in [[np.float16, 1e-2], [np.float32, 1e-2]]:
        with ops.device('cpu'):
          a = array_ops.placeholder(test_type[0], shape=[a_size, ab_size])
          b = array_ops.placeholder(test_type[0], shape=[ab_size, b_size])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r = ipu.ipu_compiler.compile(ipu_back, inputs=[a, b])

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
          self.assertAllClose(NonLinearity.approx_derivative(
              mat_value * mat_value * ab_size),
                              gradients_res_grads[0][0],
                              rtol=test_type[1])


class BeamSearchParams:
  def __init__(self, b, t, nc, bw, tp, l, s):
    self.batch_size = b
    self.max_time = t
    self.num_classes = nc
    self.beam_width = bw
    self.top_paths = tp
    self.length_init = l
    self.seed = s

  def __str__(self):
    return "BeamSearchParams: " + str(
        (self.batch_size, self.max_time, self.num_classes, self.beam_width,
         self.top_paths, self.length_init, self.seed))


def create_beam_search_params():
  return [
      BeamSearchParams(2, 10, 32, 16, 1, 3, 0),
      BeamSearchParams(2, 10, 32, 16, 1, 5, 1),
      BeamSearchParams(2, 10, 32, 16, 3, 5, 2)
  ]


class PopnnCTCLossTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @staticmethod
  def create_input_values(batch_size, max_time, num_classes, max_label_length,
                          label_length):
    labels = np.random.randint(1,
                               num_classes,
                               size=[batch_size, max_label_length])
    logits = np.random.randint(0,
                               num_classes,
                               size=[max_time, batch_size, num_classes])
    label_length = [label_length] * batch_size
    logit_length = [max_time] * batch_size

    return labels, logits, label_length, logit_length

  @staticmethod
  def logits_to_log_probs(inputs):
    inputs[1] = tf_nn_ops.log_softmax_v2(inputs[1], axis=2)
    return inputs

  @staticmethod
  def create_inputs(batch_size, max_time, num_classes, max_label_length,
                    in_dtype):
    labels = array_ops.placeholder(np.int32,
                                   shape=[batch_size, max_label_length])
    logits = array_ops.placeholder(in_dtype,
                                   shape=[max_time, batch_size, num_classes])
    label_length = array_ops.placeholder(np.int32, shape=[batch_size])
    logit_length = array_ops.placeholder(np.int32, shape=[batch_size])

    return [labels, logits, label_length, logit_length]

  @staticmethod
  def loss_and_grad(loss_function, inputs, **kwargs):
    loss = loss_function(inputs[0], inputs[1], inputs[2], inputs[3], **kwargs)
    grad = gradients_impl.gradients(loss, inputs[1])
    return loss, grad

  @staticmethod
  def create_feed_dict(inputs, input_values):
    return {
        inputs[0]: input_values[0],
        inputs[1]: input_values[1],
        inputs[2]: input_values[2],
        inputs[3]: input_values[3],
    }

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogProbs(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ops.device('cpu'):
        inputs_cpu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        loss_cpu, _ = self.loss_and_grad(ctc_ops.ctc_loss_v2,
                                         inputs_cpu,
                                         blank_index=blank_index)
        feed_dict_cpu = self.create_feed_dict(inputs_cpu, input_values)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_ipu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        feed_dict_ipu = self.create_feed_dict(inputs_ipu, input_values)
        inputs_ipu = self.logits_to_log_probs(inputs_ipu)
        loss_ipu, _ = self.loss_and_grad(
            ipu.ops.nn_ops.ctc_loss_with_log_probs,
            inputs_ipu,
            blank_index=blank_index)

      loss_value_cpu = sess.run(loss_cpu, feed_dict=feed_dict_cpu)
      loss_value_ipu = sess.run(loss_ipu, feed_dict=feed_dict_ipu)

    self.assertAllClose(loss_value_cpu, loss_value_ipu)

  @test_util.deprecated_graph_mode_only
  def testCTCGradientWithLogProbs(self):
    batch_size = 2
    label_length = 2
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 4
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ops.device('cpu'):
        inputs_cpu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        feed_dict_cpu = self.create_feed_dict(inputs_cpu, input_values)
        _, grad_cpu = self.loss_and_grad(ctc_ops.ctc_loss_v2,
                                         inputs_cpu,
                                         blank_index=blank_index)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_ipu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        feed_dict_ipu = self.create_feed_dict(inputs_ipu, input_values)
        inputs_ipu = self.logits_to_log_probs(inputs_ipu)
        _, grad_ipu = self.loss_and_grad(
            ipu.ops.nn_ops.ctc_loss_with_log_probs,
            inputs_ipu,
            blank_index=blank_index)

      grad_value_cpu = sess.run(grad_cpu, feed_dict=feed_dict_cpu)
      grad_value_ipu = sess.run(grad_ipu, feed_dict=feed_dict_ipu)

    self.assertAllClose(grad_value_cpu, grad_value_ipu)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogits(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ops.device('cpu'):
        inputs_cpu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        loss_cpu, _ = self.loss_and_grad(ctc_ops.ctc_loss_v2,
                                         inputs_cpu,
                                         blank_index=blank_index)
        feed_dict_cpu = self.create_feed_dict(inputs_cpu, input_values)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_ipu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        loss_ipu, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                                         inputs_ipu,
                                         blank_index=blank_index)
        feed_dict_ipu = self.create_feed_dict(inputs_ipu, input_values)

      loss_value_cpu = sess.run(loss_cpu, feed_dict=feed_dict_cpu)
      loss_value_ipu = sess.run(loss_ipu, feed_dict=feed_dict_ipu)

    self.assertAllClose(loss_value_cpu, loss_value_ipu)

  @test_util.deprecated_graph_mode_only
  def testCTCGradientWithLogits(self):
    batch_size = 2
    label_length = 2
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 4
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ops.device('cpu'):
        inputs_cpu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        feed_dict_cpu = self.create_feed_dict(inputs_cpu, input_values)
        _, grad_cpu = self.loss_and_grad(ctc_ops.ctc_loss_v2,
                                         inputs_cpu,
                                         blank_index=blank_index)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_ipu = self.create_inputs(batch_size, max_time, num_classes,
                                        max_label_length, np.float32)
        feed_dict_ipu = self.create_feed_dict(inputs_ipu, input_values)
        _, grad_ipu = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                                         inputs_ipu,
                                         blank_index=blank_index)

      grad_value_cpu = sess.run(grad_cpu, feed_dict=feed_dict_cpu)
      grad_value_ipu = sess.run(grad_ipu, feed_dict=feed_dict_ipu)

    self.assertAllClose(grad_value_cpu, grad_value_ipu)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogProbsOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0
    in_dtype = np.float16
    out_dtype = np.float32

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs = self.create_inputs(batch_size, max_time, num_classes,
                                    max_label_length, in_dtype)
        feed_dict = self.create_feed_dict(inputs, input_values)
        inputs = self.logits_to_log_probs(inputs)
        loss, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_with_log_probs,
                                     inputs,
                                     blank_index=blank_index,
                                     out_dtype=out_dtype)

      loss_value = sess.run(loss, feed_dict=feed_dict)
      self.assertEqual(loss_value.dtype, out_dtype)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogitsOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0
    in_dtype = np.float16
    out_dtype = np.float32

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs = self.create_inputs(batch_size, max_time, num_classes,
                                    max_label_length, in_dtype)
        feed_dict = self.create_feed_dict(inputs, input_values)
        loss, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                                     inputs,
                                     blank_index=blank_index,
                                     out_dtype=out_dtype)

      loss_value = sess.run(loss, feed_dict=feed_dict)
      self.assertEqual(loss_value.dtype, out_dtype)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogProbsDefaultOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_16 = self.create_inputs(batch_size, max_time, num_classes,
                                       max_label_length, np.float16)
        inputs_32 = self.create_inputs(batch_size, max_time, num_classes,
                                       max_label_length, np.float32)
        feed_dict_16 = self.create_feed_dict(inputs_16, input_values)
        feed_dict_32 = self.create_feed_dict(inputs_32, input_values)
        inputs_16 = self.logits_to_log_probs(inputs_16)
        inputs_32 = self.logits_to_log_probs(inputs_32)
        loss_16, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_with_log_probs,
                                        inputs_16,
                                        blank_index=blank_index)
        loss_32, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_with_log_probs,
                                        inputs_32,
                                        blank_index=blank_index)

      loss_value_16 = sess.run(loss_16, feed_dict=feed_dict_16)
      loss_value_32 = sess.run(loss_32, feed_dict=feed_dict_32)
      self.assertEqual(loss_value_16.dtype, np.float16)
      self.assertEqual(loss_value_32.dtype, np.float32)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogitsDefaultOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0

    with se.Session() as sess:
      input_values = self.create_input_values(batch_size, max_time,
                                              num_classes, max_label_length,
                                              label_length)
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        inputs_16 = self.create_inputs(batch_size, max_time, num_classes,
                                       max_label_length, np.float16)
        inputs_32 = self.create_inputs(batch_size, max_time, num_classes,
                                       max_label_length, np.float32)
        feed_dict_16 = self.create_feed_dict(inputs_16, input_values)
        feed_dict_32 = self.create_feed_dict(inputs_32, input_values)
        loss_16, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                                        inputs_16,
                                        blank_index=blank_index)
        loss_32, _ = self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                                        inputs_32,
                                        blank_index=blank_index)

      loss_value_16 = sess.run(loss_16, feed_dict=feed_dict_16)
      loss_value_32 = sess.run(loss_32, feed_dict=feed_dict_32)
      self.assertEqual(loss_value_16.dtype, np.float16)
      self.assertEqual(loss_value_32.dtype, np.float32)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogProbsInvalidOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0
    in_dtype = np.float32
    out_dtype = np.float16

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      inputs = self.create_inputs(batch_size, max_time, num_classes,
                                  max_label_length, in_dtype)
      inputs = self.logits_to_log_probs(inputs)
      with self.assertRaisesRegex(
          ValueError,
          "out_dtype cannot be float16 when dtype of data is float32"):
        self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_with_log_probs,
                           inputs,
                           blank_index=blank_index,
                           out_dtype=out_dtype)

  @test_util.deprecated_graph_mode_only
  def testCTCLossWithLogitsInvalidOutDtype(self):
    batch_size = 8
    label_length = 4
    # randomly generated labels need 2n+1 time steps because there are
    # implicit blank steps around repeated labels
    max_label_length = 2 * label_length + 1
    max_time = max_label_length
    num_classes = 8
    blank_index = 0
    in_dtype = np.float32
    out_dtype = np.float16

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      inputs = self.create_inputs(batch_size, max_time, num_classes,
                                  max_label_length, in_dtype)
      with self.assertRaisesRegex(
          ValueError,
          "out_dtype cannot be float16 when dtype of logits is float32"):
        self.loss_and_grad(ipu.ops.nn_ops.ctc_loss_v2,
                           inputs,
                           blank_index=blank_index,
                           out_dtype=out_dtype)

  @parameterized.parameters(create_beam_search_params())
  @test_util.deprecated_graph_mode_only
  def testCTCBeamSearch(self, params):
    batch_size = params.batch_size
    max_time = params.max_time
    num_classes = params.num_classes
    beam_width = params.beam_width
    top_paths = params.top_paths
    length_init = params.length_init
    seed = params.seed
    blank_index = num_classes - 1  # updstream version is always num classes - 1
    in_dtype = np.float32

    np.random.seed(seed)
    input_data = np.random.rand(max_time, batch_size, num_classes)
    signed_input_length_data = np.full([batch_size],
                                       length_init,
                                       dtype=np.int32)
    inputs = array_ops.placeholder(in_dtype,
                                   shape=[max_time, batch_size, num_classes])
    signed_input_length = array_ops.placeholder(np.int32, shape=[batch_size])

    with se.Session() as sess:
      with ipu.scopes.ipu_scope("/device:IPU:0"):

        a, b, c = ipu.ops.nn_ops.ctc_beam_search_decoder(
            inputs,
            signed_input_length,
            blank_index=blank_index,
            top_paths=top_paths,
            beam_width=beam_width,
            name="BeamSearch")

        log_probs = tf_nn_ops.log_softmax_v2(inputs, axis=2)
        d, e, f = ipu.ops.nn_ops.ctc_beam_search_decoder_with_log_probs(
            log_probs,
            signed_input_length,
            blank_index=blank_index,
            top_paths=top_paths,
            beam_width=beam_width,
            name="BeamSearch")

        # Need to wait for poplibs implmentation of ctc inference before can
        # actually call it
        probs, lengths, decoded, probs2, lengths2, decoded2 = sess.run(
            [a, b, c, d, e, f],
            feed_dict={
                inputs: input_data,
                signed_input_length: signed_input_length_data
            })

    with se.Session() as sess2:
      d, l = tf_ctc_beam_search(inputs,
                                signed_input_length,
                                beam_width=beam_width,
                                top_paths=top_paths)
      # The upstream version returns a list of sparse tensors,
      # need to change this into a dense tensor by padding it
      # with the blank index
      d = [sparse.to_dense(t, default_value=blank_index) for t in d]

      for i, t in enumerate(d):
        paddings = [[0, 0], [0, max_time - array_ops.shape(t)[1]]]
        t = array_ops.pad(t, paddings, constant_values=blank_index)
        d[i] = array_ops.reshape(t, [batch_size, 1, max_time])

      d = array_ops.concat(d, 1)

      ex_decoded, ex_probs = sess2.run(  #pylint: disable=unused-variable
          [d, l],
          feed_dict={
              inputs: input_data,
              signed_input_length: signed_input_length_data
          })

    # self consistency checks
    self.assertAllClose(probs, probs2)
    self.assertAllClose(lengths, lengths2)
    self.assertAllClose(decoded, decoded2)

    # after the length for ipu version is just junk values
    for b in range(batch_size):
      for p in range(top_paths):
        l = lengths2[b][p]
        for t in range(l, max_time):
          decoded2[b][p][t] = blank_index

    # Check against upstream version now both have same form
    self.assertAllClose(probs2, ex_probs)
    self.assertAllClose(decoded2, ex_decoded)


def gelu_cpu(features, approximate):
  if approximate:
    coeff = 0.044715
    retval = 0.5 * features * (
        1.0 + np.tanh(0.7978845608028654 *
                      (features + coeff * np.power(features, 3))))
  else:
    retval = 0.5 * features * (1.0 + math_ops.erf(
        features / math_ops.cast(1.4142135623730951, features.dtype)))

  return retval


class GeluTest(test_util.TensorFlowTestCase):
  configured = False

  def __configureIPU(self):
    if not self.configured:
      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()
      self.configured = True

  def run_gelu_test(self, n, approximate):
    with self.session() as sess:
      self.__configureIPU()

      i_h = np.linspace(-10, 10, n, dtype='float32')
      ref_h = gelu_cpu(i_h, approximate)

      with ops.device("/device:IPU:0"):
        i = array_ops.placeholder(np.float32, shape=[n])
        o = nn_ops.gelu(i, approximate)

        test_h = sess.run(o, {i: i_h})

        self.assertAllClose(ref_h, test_h)

  def testApproximateFalse(self):
    self.run_gelu_test(100, False)

  def testApproximateTrue(self):
    self.run_gelu_test(100, True)


if __name__ == "__main__":
  googletest.main()

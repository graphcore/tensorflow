# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""
Popnn primitive neural network operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from google.protobuf import json_format

from tensorflow.compiler.plugin.poplar.driver import backend_config_pb2
from tensorflow.compiler.plugin.poplar.driver import option_flag_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.ipu.ops import embedding_ops as ipu_embedding_ops


def gelu(x, approximate=True, name=None):
  """This targets the PopLibs Popnn gelu operation, optimised for execution
  on the IPU.

  Args:
    x: The input tensor.
    approximate: Use tanh()-based approximation if true, otherwise use erf()
    name: Optional op name.

  Returns:
    A `Tensor`. Has the same type the input tensor.
  """

  if approximate:
    return gen_popnn_ops.ipu_gelu(x, name=name)

  inv_sqrt_2 = 0.7071067811865475
  return 0.5 * x * (1.0 + math_ops.erf(
      x * math_ops.cast(inv_sqrt_2, x.dtype, name=name), name=name))


def hard_sigmoid(x, name=None):
  """ IPU implementation of the hard sigmoid activation function.

    Args:
    x: The input tensor.
    name: Optional op name.

  Returns:
    A `Tensor`. Has the same type the input tensor.
  """
  return gen_popnn_ops.ipu_hard_sigmoid(x, name=name)


def swish(x, name=None):
  """ IPU implementation of the swish activation function.

    Args:
    x: The input tensor.
    name: Optional op name.

  Returns:
    A `Tensor`. Has the same type the input tensor.
  """
  return gen_popnn_ops.ipu_swish(x, name=name)


def multi_conv(func=None, options=None):
  """A function decorator for generating multi-convolution operations.
  Multi-convolutions allow for a set of data-independent convolutions to be
  executed in parallel. Executing convolutions in parallel can lead to an
  increase in the data throughput.

  The ``multi_conv`` function decorator is a convenient way to generate
  multi-convolutions - it detects all the convolution operations inside of the
  decorated function and executes them in parallel.

  For example:

  .. code-block:: python

    from tensorflow import keras
    from tensorflow.python import ipu

    @ipu.nn_ops.multi_conv
    def convs(x, y, z):
      x = keras.layers.DepthwiseConv2D(8, 2, depth_multiplier=2)(x)
      y = keras.layers.DepthwiseConv2D(16, 4, depth_multiplier=2)(y)
      z = keras.layers.Conv2D(8, 3)(z)
      return x, y, z

  Will detect and execute the three convolutions ``x``, ``y`` and ``z`` in
  parallel.
  Note that any operations which are not convolutions, such as bias add
  operations, will be executed in the same way as if they were not inside of a
  ``multi_conv`` decorated function.

  It is also possible to set PopLibs multi-convolution options using this
  decorator.

  For example:

  .. code-block:: python

    from tensorflow import keras
    from tensorflow.python import ipu

    @ipu.nn_ops.multi_conv(options={"perConvReservedTiles":"50"})
    def convs(x, y, z):
      x = keras.layers.DepthwiseConv2D(8, 2, depth_multiplier=2)(x)
      y = keras.layers.DepthwiseConv2D(16, 4, depth_multiplier=2)(y)
      z = keras.layers.Conv2D(8, 3)(z)
      return x, y, z

  See the PopLibs documention for the list of all available flags.
  Note that these options will also be applied to the gradient operations
  generated during backpropagation.

  Args:
    func: A python function which takes a list of positional arguments only. All
      the arguments must be `tf.Tensor`-like objects, or be convertible to them.
      The function provided must return at least one `tf.Tensor`-like object.
    options: A dictionary of Poplar option flags for multi-convolution. See the
      multi-convolution PopLibs documentation for available flags.
  """
  def decorated(inner_func):
    def multi_conv_wrapper(*args):
      inner_options = options if options else {}

      if not isinstance(inner_options, dict):
        raise TypeError(
            "Expected the multi_conv `options` to be a `dict`, but got %s "
            "instead." % (str(inner_options)))

      option_proto = option_flag_pb2.PoplarOptionFlags()
      for key, value in inner_options.items():
        flag = option_proto.flags.add()
        flag.option = key
        flag.value = value

      def func_wrapper(*args):
        with op_util.gradient_override_scope(training=False):
          return inner_func(*args)

      args = functional_ops._convert_to_list(args)  # pylint: disable=protected-access
      with ops.name_scope("multi_conv") as scope:
        func_graph, captured_args, constant_outputs = \
          functional_ops._compile_function(  # pylint: disable=protected-access
            func_wrapper,
            args,
            scope, [],
            allow_external_captures=True)

        with ops.control_dependencies(list(func_graph.control_captures)):
          outputs = gen_functional_ops.multi_conv(
              captured_args,
              to_apply=util.create_new_tf_function(func_graph),
              Tout=func_graph.output_types,
              output_shapes=func_graph.output_shapes,
              option_flags=json_format.MessageToJson(option_proto))
          outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access

      return functional_ops._pack_sequence_as(  # pylint: disable=protected-access
          func_graph.structured_outputs, outputs)

    return multi_conv_wrapper

  if func is not None:
    return decorated(func)

  return decorated


def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            name=None,
                            seed=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.

  This is a version of the _compute_sampled_logits function in
  tensorflow/python/ops/nn_impl.py which targets the IPU-optimized embedding
  lookups.

  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).

  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
        class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.
  Returns:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    out_labels: A Tensor object with the same shape as `out_logits`.
  """
  if isinstance(weights, variables.PartitionedVariable):
    weights = list(weights)
  if not isinstance(weights, list):
    weights = [weights]

  with ops.name_scope(name, "compute_sampled_logits",
                      weights + [biases, inputs, labels]):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = ipu_embedding_ops.embedding_lookup(weights[0],
                                               math_ops.cast(
                                                   all_ids, dtypes.int32),
                                               name="weight_lookup")
    if all_w.dtype != inputs.dtype:
      all_w = math_ops.cast(all_w, inputs.dtype)

    # true_w shape is [batch_size * num_true, dim]
    true_w = array_ops.slice(
        all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))

    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = ipu_embedding_ops.embedding_lookup(biases,
                                               math_ops.cast(
                                                   all_ids, dtypes.int32),
                                               name="bias_lookup")
    if all_b.dtype != inputs.dtype:
      all_b = math_ops.cast(all_b, inputs.dtype)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(
        nn_impl._sum_rows(dots_as_matrix),  # pylint: disable=protected-access
        [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels


def sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_true=1,
                         sampled_values=None,
                         name="sampled_softmax_loss",
                         seed=None):
  """Computes and returns the sampled softmax training loss.

  This is a version of the sampled_softmax_loss function in
  tensorflow/python/ops/nn_impl.py which targets the IPU-optimized embedding
  lookup.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  A common use case is to use this method for training, and calculate the full
  softmax loss for evaluation or inference, as in the following example:

  .. code-block:: python

    if mode == "train":
      loss = tf.nn.sampled_softmax_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...)
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)

  See the TensorFlow `Candidate Sampling Algorithms Reference
  <https://www.tensorflow.org/extras/candidate_sampling.pdf>`_

  Also see Section 3 of `Jean et al., 2014 <http://arxiv.org/abs/1412.2007>`_
  (`pdf <http://arxiv.org/pdf/1412.2007.pdf>`_) for the maths.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  """
  logits, labels = _compute_sampled_logits(weights=weights,
                                           biases=biases,
                                           labels=labels,
                                           inputs=inputs,
                                           num_sampled=num_sampled,
                                           num_classes=num_classes,
                                           num_true=num_true,
                                           sampled_values=sampled_values,
                                           subtract_log_q=True,
                                           name=name,
                                           seed=seed)
  labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
  sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                               logits=logits)
  # sampled_losses is a [batch_size] tensor.
  return sampled_losses


def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             name="nce_loss"):
  """Computes and returns the noise-contrastive estimation training loss.

  This is a version of the nce_loss function in
  tensorflow/python/ops/nn_impl.py which targets the IPU-optimized embedding
  lookup.

  See `Noise-contrastive estimation: A new estimation principle for
  unnormalized statistical models
  <http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_.
  Also see the TensorFlow `Candidate Sampling Algorithms Reference
  <https://www.tensorflow.org/extras/candidate_sampling.pdf>`_.

  A common use case is to use this method for training, and calculate the full
  sigmoid loss for evaluation or inference, as in the following example:

  .. code-block:: python

    if mode == "train":
      loss = tf.nn.nce_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...)
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
      loss = tf.reduce_sum(loss, axis=1)

  Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
  so your labels must be sorted in order of decreasing frequency to achieve
  good results.  For more details, see
  `tf.random.log_uniform_candidate_sampler`.

  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.

  Note: It would be useful to allow a variable number of target classes per
  example.  TensorFlow hopes to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of negative classes to randomly sample
        per batch. This single sample of negative classes is evaluated for each
        element in the batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.
  """
  logits, labels = _compute_sampled_logits(weights=weights,
                                           biases=biases,
                                           labels=labels,
                                           inputs=inputs,
                                           num_sampled=num_sampled,
                                           num_classes=num_classes,
                                           num_true=num_true,
                                           sampled_values=sampled_values,
                                           subtract_log_q=True,
                                           name=name)
  sampled_losses = nn_impl.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name="sampled_losses")
  # sampled_losses is batch_size x {true_loss, sampled_losses...}
  # We sum out true and sampled losses.
  return nn_impl._sum_rows(sampled_losses)  # pylint: disable=protected-access


def ctc_loss_v2(labels,
                logits,
                label_length,
                logit_length,
                blank_index,
                out_dtype=None,
                name=None):
  """Calculates and returns CTC (Connectionist Temporal Classification) loss.
  This op is designed and optimized for the IPU and cannot be used with other
  systems.

  Note: The TensorFlow op tf.nn.ctc_loss is not compatible with the IPU.

  Args:
    labels: The labels input [batch_size, max_label_length] tensor.
    logits: The data input [max_time, batch_size, num_classes] tensor.
        The data is expected in the form of logits.
    label_length: A tensor of shape [batch_size] containing the number of
        labels in each `labels` batch entry.
    logit_length: A tensor of shape [batch_size] containing the number of
        timesteps in each `logits` batch entry.
    blank_index: The class index to use for the blank label.
    out_dtype: The dtype of the loss tensor (float16 or float32).
        Cannot be float16 if the dtype of `logits` is float32.
        Default: the same dtype as `logits`.
    name: A name for this op. Defaults to "ctc_loss".

  Returns:
    A loss tensor of shape [batch_size].
  """
  if out_dtype is None:
    out_dtype = logits.dtype
  elif logits.dtype == dtypes.float32 and out_dtype == dtypes.float16:
    raise ValueError(
        "out_dtype cannot be float16 when dtype of logits is float32.")

  loss, _ = gen_popnn_ops.popnn_ctc_loss_with_logits(logits,
                                                     labels,
                                                     logit_length,
                                                     label_length,
                                                     out_dtype=out_dtype,
                                                     blank_index=blank_index,
                                                     name=name)

  return loss


def ctc_loss_with_log_probs(labels,
                            data,
                            label_length,
                            data_length,
                            blank_index,
                            out_dtype=None,
                            name=None):
  """Calculates and returns CTC (Connectionist Temporal Classification) loss.
  This op is designed and optimized for the IPU and cannot be used with other
  systems. It is identical to the :py:func:`ctc_loss_v2` operation except that
  it takes negative log probabilities instead of logits for the data input.

  Note: The TensorFlow op tf.nn.ctc_loss is not compatible with the IPU.

  Args:
    labels: The labels input [batch_size, max_label_length] tensor.
    data: The data input [max_time, batch_size, num_classes] tensor.
        The data is expected in the form of log probabilities.
    label_length: A tensor of shape [batch_size] containing the number of
        labels in each `labels` batch entry.
    data_length: A tensor of shape [batch_size] containing the number of
        timesteps in each `data` batch entry.
    blank_index: The class index to use for the blank label.
    out_dtype: The dtype of the loss tensor.
        Cannot be float16 if the dtype of `data` is float32.
        Default: the same dtype as `data`.
    name: A name for this op. Defaults to "ctc_loss".

  Returns:
    A loss tensor of shape [batch_size].
  """
  if out_dtype is None:
    out_dtype = data.dtype
  elif data.dtype == dtypes.float32 and out_dtype == dtypes.float16:
    raise ValueError(
        "out_dtype cannot be float16 when dtype of data is float32.")

  loss, _ = gen_popnn_ops.popnn_ctc_loss_with_log_probs(
      data,
      labels,
      data_length,
      label_length,
      out_dtype=out_dtype,
      blank_index=blank_index,
      name=name)

  return loss


def ctc_beam_search_decoder(logits,
                            logits_lengths,
                            beam_width=100,
                            top_paths=1,
                            blank_index=-1,
                            name=None):
  """Calculates and returns CTC (Connectionist Temporal Classification)
  predictions.
  This op is designed and optimized for the IPU and cannot be used with other
  systems.

  .. code-block:: python

    # assuming batch_size = 1
    # hyper-parameters
    top_paths = 1
    beam_width = 100

    if mode == "predict":

      probs, lengths, predictions = ctc_beam_search_decoder(logits,
                                                            logits_lengths,
                                                            beam_width,
                                                            top_paths)

      batch_index = 0 # as batch_size 1, otherwise must iterate batch
      path_index = 0 # as top_paths = 1 otherwise argmin(probs[batch_index])

      vocab_predictions = [tokens[predictions[batch_index][path_index][l]] for l
                                 in range(lengths[batch_index)]
      predicted_prob_of_correct_prediction = probs[batch_index][path_index]
      return vocab_predictions, predicted_prob_of_correct_prediction

  Note: The TensorFlow op tf.nn.ctc_beam_search_decoder is not
  compatible with the IPU. This version also returns the predicted
  label lengths in addition to the probabilities and decoded labels.
  Instead of returning a lengths tensor the upstream version returns
  a list of dynamically sized tensors.

  Args:
    logits: The data input [max_time, batch_size, num_classes] tensor.
        The data is expected in the form of logits.
    logit_lengths: A tensor of shape [batch_size] containing the number of
        valid timesteps in each `logits` batch entry.
    beam_width: The beam width to be passed to the beam search algorithm.
    top_paths: The number of paths to keep track of in the beam
        search algorithm. This must be less than or equal to `beam_width`.
    blank_index: The class index to use for the blank label.
    name: A name for this op. Defaults to "ctc_beam_search".

  Returns:

    * A tensor of shape [batch_size, top_paths] containing the negative log
      probabilities of the `top_paths` most likely labels.
    * A tensor of shape [batch_size, top_paths] containing the length of the
      `top_paths` most likely labels.
    * A tensor of shape [batch_size, top_paths, max_time] containing the
      decoded `top_paths` most likely labels.

  """

  label_probabilities, label_lengths, decoded_labels =\
    gen_popnn_ops.popnn_ctc_beam_search_with_logits(
        logits,
        logits_lengths,
        blank_index=blank_index,
        top_paths=top_paths,
        beam_width=beam_width,
        name=name)
  return label_probabilities, label_lengths, decoded_labels


def ctc_beam_search_decoder_with_log_probs(log_probs,
                                           input_lengths,
                                           beam_width=100,
                                           top_paths=1,
                                           blank_index=-1,
                                           name=None):
  """Calculates and returns CTC (Connectionist
  Temporal Classification) predictions.
  This op is designed and optimized for the IPU and cannot be used with other
  systems. It is identical to the :py:func:`ctc_beam_search_decoder`
  operation except that it takes negative log
  probabilities instead of logits for the data input.

  Note: The TensorFlow op tf.nn.beam_search_decoder is not
  compatible with the IPU. This version also returns the predicted
  label lengths in addition to the probabilities and decoded labels.

  Args:
    log_probs: The data input [max_time, batch_size, num_classes] tensor.
        The data is expected in the form of log probabilities.
    input_lengths: A tensor of shape [batch_size] containing the number of
        valid timesteps in each `log_probs` batch entry.
    beam_width: The beam width to be passed to the beam search algorithm.
    top_paths: The number of paths to keep track of in the beam
        search algorithm. This must be less than or equal to `beam_width`.
    blank_index: The class index to use for the blank label.
    name: A name for this op. Defaults to "ctc_beam_search".

  Returns:

    * A tensor of shape [batch_size, top_paths] containing the negative log
      probabilities of the `top_paths` most likely labels.
    * A tensor of shape [batch_size, top_paths] containing the length of the
      `top_paths` most likely labels.
    * A tensor of shape [batch_size, top_paths, max_time] containing the
      decoded `top_paths` most likely labels.

  """
  label_probabilities, label_lengths, decoded_labels =\
    gen_popnn_ops.popnn_ctc_beam_search_with_log_probs(
        log_probs,
        input_lengths,
        blank_index=blank_index,
        top_paths=top_paths,
        beam_width=beam_width,
        name=name)
  return label_probabilities, label_lengths, decoded_labels

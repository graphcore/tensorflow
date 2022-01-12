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

import os
import tempfile
from functools import partial, reduce
import threading
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn, nn_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops, dtypes
from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.ipu import ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import rand_ops
from tensorflow.python.ipu import config
from tensorflow.python.ipu.ops import application_compile_op
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python.training import momentum
from tensorflow.python.keras.datasets import mnist
from tensorflow.compat.v1 import train
from tensorflow.python.ipu import embedded_runtime
from tensorflow.python.ipu import pipelining_ops

ops.disable_eager_execution()
disable_v2_behavior()

L1_SIZE = 320
L2_SIZE = 72
L3_SIZE = 10
NUM_PIXELS = 784
BATCH_SIZE = 16
NUM_ITERATIONS = 8
NUM_ENGINE_ITERATIONS = 4

NUM_ENGINES = 2
NUM_THREADS_PER_ENGINE = 4
NUM_SESSIONS = 4

NUM_TEST_ITERATIONS = NUM_ENGINES * NUM_SESSIONS * NUM_THREADS_PER_ENGINE \
  * NUM_ENGINE_ITERATIONS * NUM_ITERATIONS
NUM_OUTER_ITERATIONS = NUM_ENGINES * NUM_SESSIONS * NUM_THREADS_PER_ENGINE \
  * NUM_ENGINE_ITERATIONS


def dense_layer(hiddenSize, input_, scope_name):
  with variable_scope.variable_scope(scope_name,
                                     reuse=variable_scope.AUTO_REUSE,
                                     use_resource=True):
    w = variable_scope.get_variable(
        "weight",
        shape=[input_.shape[-1], hiddenSize],
        initializer=init_ops.glorot_uniform_initializer())
    b = variable_scope.get_variable(
        "bias",
        shape=[hiddenSize],
        initializer=init_ops.glorot_uniform_initializer())
    return nn.relu_layer(input_, w, b)


def test_model(outqueue, inputs):
  relu1 = dense_layer(L1_SIZE, inputs, "d1")
  relu2 = dense_layer(L2_SIZE, relu1, "d2")
  relu3 = dense_layer(L3_SIZE, relu2, "d3")

  return outqueue.enqueue({'predictions': relu3})


def test_model_pipelined(infeed_queue, outfeed_queue):
  pipeline_depth = 2

  def stage1(images):
    relu1 = dense_layer(L1_SIZE, images, "d1")
    return relu1

  def stage2(relu1):
    relu2 = dense_layer(L2_SIZE, relu1, "d2")
    relu3 = dense_layer(L3_SIZE, relu2, "d3")
    return relu3

  return pipelining_ops.pipeline(
      [stage1, stage2],
      gradient_accumulation_count=pipeline_depth,
      repeat_count=NUM_ITERATIONS / pipeline_depth,
      infeed_queue=infeed_queue,
      outfeed_queue=outfeed_queue,
      pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)


def loop_builder(iterations, builder_func, infeed):
  return loops.repeat(iterations, builder_func, [], infeed)


def run_and_export_model(tmp_dir,
                         poplar_exec_output_path,
                         pipelined,
                         freeze_variables=True,
                         images=None):
  n_test = BATCH_SIZE * NUM_TEST_ITERATIONS
  if images is None:
    images = np.random.rand(n_test, NUM_PIXELS).astype(np.float32)

  test_dataset = dataset_ops.Dataset.from_tensor_slices((images,))
  test_dataset = test_dataset.cache().repeat().batch(BATCH_SIZE,
                                                     drop_remainder=True)

  infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(test_dataset)
  outfeed_test_queue = ipu_outfeed_queue.IPUOutfeedQueue()

  if pipelined:
    bound_test_loop = partial(test_model_pipelined, infeed_test_queue,
                              outfeed_test_queue)
  else:
    bound_test_model = partial(test_model, outfeed_test_queue)
    bound_test_loop = partial(loop_builder, NUM_ITERATIONS, bound_test_model,
                              infeed_test_queue)

  # Use the bound builder functions to place the model on the IPU:
  with scopes.ipu_scope("/device:IPU:0"):
    test_loop = ipu_compiler.compile(bound_test_loop)

  # Initialisers should go on the CPU:
  with ops.device("cpu"):
    saver = train.Saver()

  # Setup and acquire an IPU device:
  cfg = config.IPUConfig()
  cfg.auto_select_ipus = 2 if pipelined else 1
  tu.add_hw_ci_connection_options(cfg)
  cfg.configure_ipu_system()

  # These allow us to retrieve the results of IPU feeds:
  dequeue_test_outfeed = outfeed_test_queue.dequeue()

  # Run the model:
  with sl.Session() as sess:
    sess.run(variables.global_variables_initializer())

    model_save_path = f'{tmp_dir}/model'
    saver.save(sess, model_save_path)

    print(f"  Testing...")

    output = np.empty(
        [NUM_OUTER_ITERATIONS, NUM_ITERATIONS, BATCH_SIZE, L3_SIZE],
        dtype='float32')

    sess.run(infeed_test_queue.initializer)

    for ei in range(NUM_OUTER_ITERATIONS):
      sess.run(test_loop)
      result = sess.run(dequeue_test_outfeed,)
      if pipelined:
        output[ei, :, :, :] = result[0] if versions.VERSION.startswith(
            '1') else result
      else:
        output[ei, :, :, :] = result['predictions']

    d1_bias = train.load_variable(model_save_path, 'd1/bias')
    d1_weight = train.load_variable(model_save_path, 'd1/weight')
    d2_bias = train.load_variable(model_save_path, 'd2/bias')
    d2_weight = train.load_variable(model_save_path, 'd2/weight')
    d3_bias = train.load_variable(model_save_path, 'd3/bias')
    d3_weight = train.load_variable(model_save_path, 'd3/weight')

    model_ref = dict(d1_bias=d1_bias,
                     d1_weight=d1_weight,
                     d2_bias=d2_bias,
                     d2_weight=d2_weight,
                     d3_bias=d3_bias,
                     d3_weight=d3_weight,
                     images=images,
                     output=output)

  # Use a new graph and session for the compilation.
  with ops.Graph().as_default(), sl.Session() as sess:
    compile_op = application_compile_op.experimental_application_compile_op(
        bound_test_loop,
        output_path=poplar_exec_output_path,
        freeze_variables=freeze_variables)

    # Load the weights into the new session.
    train.Saver().restore(sess, model_save_path)

    print(f"  Compiling and exporting...")
    sess.run(compile_op)

  config.reset_ipu_configuration()

  return model_ref


def _build_executable(tmp_dir_obj,
                      pipelined=False,
                      freeze_variables=True,
                      poplar_exec_filepath=None,
                      images=None):
  tmp_dir = tmp_dir_obj.name

  if poplar_exec_filepath is None:
    poplar_exec_filepath = os.path.join(tmp_dir, "application.poplar_exec")

  model_ref = run_and_export_model(tmp_dir,
                                   poplar_exec_filepath,
                                   pipelined,
                                   freeze_variables=freeze_variables,
                                   images=images)

  return (model_ref, poplar_exec_filepath)


TESTCASES = [{
    'testcase_name':
    (f'_pipelined_{pipelined}_multiple_sessions_{multiple_sessions}_' +
     f'multiple_threads_{multiple_threads}_' +
     f'multiple_engines_{multiple_engines}'),
    'pipelined':
    pipelined,
    'multiple_sessions':
    multiple_sessions,
    'multiple_threads':
    multiple_threads,
    'multiple_engines':
    multiple_engines,
} for pipelined in [False, True] for multiple_sessions in [False, True]
             for multiple_threads in [False, True]
             for multiple_engines in [False, True]]


class ApplicationRuntimeTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  reference_cache_initiazed = False
  initializer_lock = threading.Lock()

  @staticmethod
  def __init_reference_data():
    with ApplicationRuntimeTest.initializer_lock:
      if not ApplicationRuntimeTest.reference_cache_initiazed:
        ApplicationRuntimeTest.tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir_obj = ApplicationRuntimeTest.tmp_dir_obj
        tmp_dir = tmp_dir_obj.name

        ApplicationRuntimeTest.non_pipelined_poplar_exec_filepath = \
          os.path.join(tmp_dir,
                       "non_pipelined_application.poplar_exec")
        ApplicationRuntimeTest.pipelined_poplar_exec_filepath = \
          os.path.join(tmp_dir,
                       "pipelined_application.poplar_exec")

        n_test = NUM_TEST_ITERATIONS * BATCH_SIZE
        ApplicationRuntimeTest.images = np.random.rand(
            n_test, NUM_PIXELS).astype(np.float32)
        images = ApplicationRuntimeTest.images

        ApplicationRuntimeTest.non_pipelined_model_ref, _ = _build_executable(
            tmp_dir_obj,
            pipelined=False,
            freeze_variables=True,
            poplar_exec_filepath=ApplicationRuntimeTest.
            non_pipelined_poplar_exec_filepath,
            images=images)

        ApplicationRuntimeTest.pipelined_model_ref, _ = _build_executable(
            tmp_dir_obj,
            pipelined=True,
            freeze_variables=True,
            poplar_exec_filepath=ApplicationRuntimeTest.
            pipelined_poplar_exec_filepath,
            images=images)

        ApplicationRuntimeTest.reference_cache_initiazed = True

  @parameterized.named_parameters(*TESTCASES)
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test(self, pipelined, multiple_sessions, multiple_threads,
           multiple_engines):
    ApplicationRuntimeTest.__init_reference_data()

    if (multiple_sessions or multiple_engines
        or pipelined) and not multiple_threads:
      return

    if multiple_engines and not multiple_sessions:
      return

    if pipelined:
      poplar_exec_filepath = \
        ApplicationRuntimeTest.pipelined_poplar_exec_filepath
      ref_output = ApplicationRuntimeTest.pipelined_model_ref['output']
    else:
      poplar_exec_filepath = \
        ApplicationRuntimeTest.non_pipelined_poplar_exec_filepath
      ref_output = ApplicationRuntimeTest.non_pipelined_model_ref['output']

    images = ApplicationRuntimeTest.images

    num_engines = NUM_ENGINES if multiple_engines else 1
    num_threads_per_engine = NUM_THREADS_PER_ENGINE if multiple_threads else 1

    num_threads = num_engines * num_threads_per_engine

    images_ph = array_ops.placeholder(dtypes.float32,
                                      shape=[BATCH_SIZE, NUM_PIXELS],
                                      name='images')

    test_shape = (num_threads, NUM_ENGINE_ITERATIONS, NUM_ITERATIONS)
    n_test = reduce(lambda p, x: p * x, test_shape)

    images_local = images.reshape((-1, BATCH_SIZE, NUM_PIXELS))
    images_local = images_local[0:n_test, :, :]
    images_local = images_local.reshape(*(test_shape +
                                          (BATCH_SIZE, NUM_PIXELS)))

    ref_output = ref_output.reshape((-1, BATCH_SIZE, L3_SIZE))
    ref_output = ref_output[0:n_test, :, :]
    ref_output = ref_output.reshape(*(test_shape + (BATCH_SIZE, L3_SIZE)))

    output = np.empty(ref_output.shape, dtype='float32')

    engine_name_prefix = f'engine_pipelined_{pipelined}'

    def run_loops(sess, result, infeeds, t):
      for ei in range(NUM_ENGINE_ITERATIONS):
        for li in range(NUM_ITERATIONS):
          images_host = images_local[t, ei, li, :, :]

          results = sess.run(
              result,
              feed_dict={
                  infeeds: (images_host,),  # pylint: disable=cell-var-from-loop
              })
          output[t, ei, li, :, :] = results[0]

    def inference_thread(sess, res, infeeds_, t):
      if multiple_engines:
        engine_name = f'{engine_name_prefix}_{t % NUM_ENGINES}'
      else:
        engine_name = engine_name_prefix

      if multiple_sessions:
        with sl.Session() as session:
          run_app = gen_application_runtime.application_runtime(
              inputs=(),
              filename=poplar_exec_filepath,
              engine_name=engine_name)

          images_ph = array_ops.placeholder(dtypes.float32,
                                            shape=[BATCH_SIZE, NUM_PIXELS],
                                            name='images')

          infeeds = (images_ph,)
          result = gen_application_runtime.application_call(
              infeeds,
              anchor=run_app,
              outfeed_types=[dtypes.float32],
              engine_name=engine_name)

          session.graph.finalize()
          run_loops(session, result, infeeds, t)
      else:
        with sess.graph.as_default():
          with sess.as_default():
            if multiple_engines:
              run_app = gen_application_runtime.application_runtime(
                  inputs=(),
                  filename=poplar_exec_filepath,
                  engine_name=engine_name)

              images_ph = array_ops.placeholder(dtypes.float32,
                                                shape=[BATCH_SIZE, NUM_PIXELS],
                                                name=f'images')

              infeeds = (images_ph,)
              result = gen_application_runtime.application_call(
                  infeeds,
                  anchor=run_app,
                  outfeed_types=[dtypes.float32],
                  engine_name=engine_name)

              run_loops(sess, result, infeeds, t)
            else:
              run_loops(sess, res, infeeds_, t)

    def run_across_threads(session=None, result=None, infeeds=None):
      thread_list = []
      for t in range(num_threads):
        if multiple_threads:
          thread = threading.Thread(target=inference_thread,
                                    args=(session, result, infeeds, t))
          thread_list.append(thread)
          thread.start()
        else:
          inference_thread(session, result, infeeds, t)

      for thread in thread_list:
        thread.join()

    if multiple_sessions:
      run_across_threads()
    elif multiple_engines:
      with sl.Session() as session:
        run_across_threads(session, None, None)
    else:
      with sl.Session() as session:
        run_app = gen_application_runtime.application_runtime(
            inputs=(),
            filename=poplar_exec_filepath,
            engine_name=engine_name_prefix)

        images_ph = array_ops.placeholder(dtypes.float32,
                                          shape=[BATCH_SIZE, NUM_PIXELS],
                                          name='images')

        infeeds = (images_ph,)
        result = gen_application_runtime.application_call(
            infeeds,
            anchor=run_app,
            outfeed_types=[dtypes.float32],
            engine_name=engine_name_prefix)

        session.graph.finalize()
        run_across_threads(session, result, infeeds)

    self.assertAllClose(ref_output, output)


class EmbeddedRuntimeTest(test_util.TensorFlowTestCase,
                          parameterized.TestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_wrapper(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()

    model_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        pipelined=False,
                                                        freeze_variables=False)

    input_descs = [
        ('XLA_Args/d1/weight', [NUM_PIXELS, L1_SIZE], dtypes.float32, 0),
        ('XLA_Args/d1/bias', [L1_SIZE], dtypes.float32, 1),
        ('XLA_Args/d2/weight', [L1_SIZE, L2_SIZE], dtypes.float32, 2),
        ('XLA_Args/d2/bias', [L2_SIZE], dtypes.float32, 3),
        ('XLA_Args/d3/weight', [L2_SIZE, L3_SIZE], dtypes.float32, 4),
        ('XLA_Args/d3/bias', [L3_SIZE], dtypes.float32, 5),
    ]

    inputs = {
        'XLA_Args/d1/weight': model_ref['d1_weight'],
        'XLA_Args/d1/bias': model_ref['d1_bias'],
        'XLA_Args/d2/weight': model_ref['d2_weight'],
        'XLA_Args/d2/bias': model_ref['d2_bias'],
        'XLA_Args/d3/weight': model_ref['d3_weight'],
        'XLA_Args/d3/bias': model_ref['d3_bias'],
    }

    input_placeholders = []
    input_list = [None] * len(input_descs)
    for name, shape, dtype, order in input_descs:
      input_ph = array_ops.placeholder(dtype, shape=shape, name=name)
      input_placeholders.append(input_ph)
      input_list[order] = inputs[name]

    input_tuple = tuple(input_list)

    input_placeholders = tuple(input_placeholders)

    n_test = NUM_TEST_ITERATIONS

    images = array_ops.placeholder(dtypes.float32,
                                   shape=[BATCH_SIZE, NUM_PIXELS],
                                   name='images')

    images_all = model_ref['images'].reshape((-1, BATCH_SIZE, NUM_PIXELS))
    images_all = images_all[0:n_test, :, :]

    labels_all = np.ones([n_test, BATCH_SIZE, L3_SIZE], dtype='float32')

    labels_ref = model_ref['output'].reshape((-1, BATCH_SIZE, L3_SIZE))
    labels_ref = labels_ref[0:n_test, :, :]

    with sl.Session() as session:
      engine_name = f'engine_{self.id()}'

      ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath,
                                                    inputs, engine_name)

      infeeds = (images,)
      result = embedded_runtime.embedded_runtime_call(infeeds, ctx)

      for j in range(n_test):
        images_host = images_all[j, :, :]

        results = session.run(result,
                              feed_dict={
                                  infeeds: (images_host,),
                                  input_placeholders: input_tuple,
                              })
        labels_all[j, :, :] = results[0]

    self.assertAllClose(labels_ref, labels_all)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_input_error(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    model_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        pipelined=False,
                                                        freeze_variables=False)

    inputs = {
        'XLA_Args/d1/bias': model_ref['d1_bias'],
        'XLA_Args/d1/weight': model_ref['d1_weight'],
        'XLA_Args/d2/bias': model_ref['d2_bias'],
        'XLA_Args/d3/bias': model_ref['d3_bias'],
        'XLA_Args/d3/weight': model_ref['d3_weight'],
    }

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(
          Exception,
          "Failed to find input tensor with name 'XLA_Args/d2/weight' in "
          "input dictionary."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_no_list(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    _, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                freeze_variables=False)

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(Exception,
                                  "Expected the inputs to be a list."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, 4,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_too_many_inputs(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    mnist_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        freeze_variables=False)

    inputs = [
        mnist_ref['d1_bias'], mnist_ref['d1_weight'], mnist_ref['d2_bias'],
        mnist_ref['d2_weight'], mnist_ref['d3_bias'], mnist_ref['d1_weight'],
        mnist_ref['d1_weight']
    ]

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(
          Exception,
          "Embedded application runtime expects 6 inputs, but 7 were "
          "provided."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_too_few_inputs(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    mnist_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        freeze_variables=False)

    inputs = [
        mnist_ref['d1_bias'], mnist_ref['d1_weight'], mnist_ref['d2_bias']
    ]

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(
          Exception,
          "Embedded application runtime expects 6 inputs, but 3 were "
          "provided."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_wrong_shape(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    model_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        freeze_variables=False)

    inputs = {
        'XLA_Args/d1/weight': model_ref['d1_weight'],
        'XLA_Args/d1/bias': model_ref['d1_weight'],
        'XLA_Args/d2/weight': model_ref['d2_weight'],
        'XLA_Args/d2/bias': model_ref['d2_bias'],
        'XLA_Args/d3/weight': model_ref['d3_weight'],
        'XLA_Args/d3/bias': model_ref['d3_bias'],
    }

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(
          Exception,
          "Mismatched input shape at position 1 \\('XLA_Args/d1/bias'\\). "
          "Expected \\[320\\], but input 1 has shape \\[784, 320\\]."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_wrong_type(self):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    mnist_ref, poplar_exec_filepath = _build_executable(tmp_dir_obj,
                                                        freeze_variables=False)

    inputs = [
        mnist_ref['d1_weight'],
        mnist_ref['d2_bias'],
        mnist_ref['d2_weight'],
        mnist_ref['d3_bias'],
        mnist_ref['d3_weight'],
    ]

    inputs = {
        'XLA_Args/d1/weight': mnist_ref['d1_weight'],
        'XLA_Args/d1/bias': np.ones((320), dtype=np.int32),
        'XLA_Args/d2/weight': mnist_ref['d2_weight'],
        'XLA_Args/d2/bias': mnist_ref['d2_bias'],
        'XLA_Args/d3/weight': mnist_ref['d3_weight'],
        'XLA_Args/d3/bias': mnist_ref['d3_bias'],
    }

    with sl.Session():
      engine_name = f'engine_{self.id()}'

      with self.assertRaisesRegex(
          Exception,
          "Mismatched input dtype at position 1 \\('XLA_Args/d1/bias'\\). "
          "Expected <dtype: 'float32'>, but input 1 has dtype int32."):
        embedded_runtime.embedded_runtime_start(poplar_exec_filepath, inputs,
                                                engine_name)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def test_pipeline_flush(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x):
      return x @ constant_op.constant(1.0, shape=[2, 2])

    def stage2(x):
      return math_ops.reduce_sum(x)

    def my_net():
      return pipelining_ops.pipeline([stage1, stage2],
                                     12,
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    with tu.ipu_session() as sess:
      cfg = config.IPUConfig()
      cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      with tempfile.TemporaryDirectory() as tmp_dir:
        poplar_exec_filepath = os.path.join(tmp_dir, "application.poplar_exec")

        compile_op = application_compile_op.experimental_application_compile_op(
            my_net, output_path=poplar_exec_filepath)
        sess.run(compile_op)
        config.reset_ipu_configuration()

        ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                      "pipeline_flush")
        input_data = array_ops.placeholder(np.float32, shape=[2, 2])
        result = embedded_runtime.embedded_runtime_call([input_data], ctx)
        outputs1 = sess.run(
            result,
            feed_dict={input_data: np.full([2, 2], 1.0, dtype=np.float32)})
        self.assertAllClose(outputs1[0], 8.)
        outputs2 = sess.run(
            result,
            feed_dict={input_data: np.full([2, 2], 2.0, dtype=np.float32)})
        self.assertAllClose(outputs2[0], 16.)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_io_overlap_flush(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      x = x @ constant_op.constant(1.0, shape=[2, 2])
      x = math_ops.reduce_sum(x)
      return outfeed_queue.enqueue(x)

    def my_net():
      return loops.repeat(10, body, [], infeed_queue)

    with tu.ipu_session() as sess:
      cfg = config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.io_tiles.num_io_tiles = 32
      cfg.io_tiles.place_ops_on_io_tiles = True
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      with tempfile.TemporaryDirectory() as tmp_dir:
        poplar_exec_filepath = os.path.join(tmp_dir, "application.poplar_exec")

        compile_op = application_compile_op.experimental_application_compile_op(
            my_net, output_path=poplar_exec_filepath)
        sess.run(compile_op)
        config.reset_ipu_configuration()

        ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                      "io_overlap_flush")
        input_data = array_ops.placeholder(np.float32, shape=[2, 2])
        result = embedded_runtime.embedded_runtime_call([input_data], ctx)
        outputs1 = sess.run(
            result,
            feed_dict={input_data: np.full([2, 2], 1.0, dtype=np.float32)})
        self.assertAllClose(outputs1[0], 8.)
        outputs2 = sess.run(
            result,
            feed_dict={input_data: np.full([2, 2], 2.0, dtype=np.float32)})
        self.assertAllClose(outputs2[0], 16.)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_multiple_infeeds(self):
    a = np.array([2, 3], dtype='float32')
    b = np.array([4, 5], dtype='float32')
    c = np.array([6, 7], dtype='float32')

    ds = dataset_ops.Dataset.from_tensor_slices((a, b, c))
    ds = ds.cache().repeat().batch(2, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net(outq, a, b, c):
      return outq.enqueue({'result': a * b + c})

    model = partial(my_net, outfeed_queue)
    model = partial(loop_builder, 1, model, infeed_queue)

    a_ph = array_ops.placeholder(dtypes.float32, shape=(2), name='a')
    b_ph = array_ops.placeholder(dtypes.float32, shape=(2), name='b')
    c_ph = array_ops.placeholder(dtypes.float32, shape=(2), name='c')

    with tu.ipu_session() as sess, tempfile.TemporaryDirectory() as tmp_dir:
      cfg = config.IPUConfig()
      cfg.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      poplar_exec_filepath = os.path.join(
          tmp_dir, f'application_{self.id()}.poplar_exec')

      compile_op = application_compile_op.experimental_application_compile_op(
          model, output_path=poplar_exec_filepath)
      sess.run(compile_op)
      config.reset_ipu_configuration()

      ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                    "multiple_infeeds")
      result = embedded_runtime.embedded_runtime_call([a_ph, b_ph, c_ph], ctx)
      outputs = sess.run(result, feed_dict={a_ph: a, b_ph: b, c_ph: c})

      self.assertAllClose(outputs[0], [14, 22])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_embedded_runtime_exception(self):
    # The dataset for feeding the graphs.
    ds = dataset_ops.Dataset.from_tensors(constant_op.constant(1.0, shape=[1]))
    ds = ds.repeat()

    # The host side queues.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      return outfeed_queue.enqueue(12.0 / x)

    # Wrap in a loop.
    def my_net():
      r = loops.repeat(16, body, [], infeed_queue)
      return r

    def exception_executable(tmp_dir):
      poplar_exec_filepath = os.path.join(tmp_dir.name,
                                          "application.poplar_exec")

      cfg = config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.floating_point_behaviour.div0 = True
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      # Compile the application.
      compile_op = application_compile_op.experimental_application_compile_op(
          my_net, output_path=poplar_exec_filepath)
      with sl.Session() as sess:
        sess.run(compile_op)
      config.reset_ipu_configuration()

      return poplar_exec_filepath

    tmp_dir_obj = tempfile.TemporaryDirectory()
    poplar_exec_filepath = exception_executable(tmp_dir_obj)

    engine_name = f'engine_{self.id()}'
    ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                  engine_name)
    result = embedded_runtime.embedded_runtime_call(
        [np.zeros((1), dtype=np.float32)], ctx)

    with sl.Session() as sess:
      with self.assertRaisesRegex(
          errors.InternalError,
          r"\[Poplar\]\[Execute engine\] application_runtime_error: \[Recovery "
          r"action: IPU_RESET\] Tiles in excepted state [\s\S]* IPU will be "
          r"reset the next time a program is executed."):
        sess.run(result)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_reset_engine(self):
    # The dataset for feeding the graphs.
    ds = dataset_ops.Dataset.from_tensors(constant_op.constant(1.0, shape=[1]))
    ds = ds.repeat()

    # The host side queues.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      return outfeed_queue.enqueue(12.0 / x)

    # Wrap in a loop.
    def my_net():
      r = loops.repeat(16, body, [], infeed_queue)
      return r

    def exception_executable(tmp_dir):
      poplar_exec_filepath = os.path.join(tmp_dir.name,
                                          "application.poplar_exec")

      cfg = config.IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.floating_point_behaviour.div0 = True
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      # Compile the application.
      compile_op = application_compile_op.experimental_application_compile_op(
          my_net, output_path=poplar_exec_filepath)
      with sl.Session() as sess:
        sess.run(compile_op)
      config.reset_ipu_configuration()

      return poplar_exec_filepath

    tmp_dir_obj = tempfile.TemporaryDirectory()
    poplar_exec_filepath = exception_executable(tmp_dir_obj)

    engine_name = f'engine_{self.id()}'
    ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                  engine_name)
    input_data = array_ops.placeholder(np.float32, shape=[1])
    result = embedded_runtime.embedded_runtime_call([input_data], ctx)

    with sl.Session() as sess:
      # Second to last execution will fail, the last execution should pass.
      inputs = [1.0] * 3 + [0.0, 1.0]
      failures = [False] * 3 + [True, False]
      for val, should_fail in zip(inputs, failures):
        failed = False
        try:
          x = sess.run(result, {input_data: [val]})
          self.assertEqual(x, [12.0])
        except:  # pylint: disable=bare-except
          failed = True
        self.assertEqual(failed, should_fail)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_large_input_count(self):
    loop_count = 16

    ds = dataset_ops.Dataset.from_tensors(
        tuple([
            np.ones(()).astype(np.float32),
            np.ones((1, 5)).astype(np.float32),
            np.ones((5, 1)).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
            np.ones(()).astype(np.float32),
        ]))
    ds = ds.repeat()

    # The host side queues.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    # The device side main.
    def body(inputs_0, inputs_1, inputs_2, inputs_3, inputs_4, inputs_5,
             inputs_6, inputs_7, inputs_8, inputs_9, inputs_10, inputs_11):
      result = inputs_0
      result = result + math_ops.matmul(inputs_1, inputs_2)
      result = result + inputs_3
      result = result + inputs_4
      result = result + inputs_5
      result = result + inputs_6
      result = result + inputs_7
      result = result + inputs_8
      result = result + inputs_9
      result = result + inputs_10
      result = result + inputs_11

      outfeed = outfeed_queue.enqueue({'result': result})
      return outfeed

    # Wrap in a loop.
    def my_net():
      r = loops.repeat(loop_count, body, [], infeed_queue)
      return r

    # Configure the IPU for compilation.
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    # Setup a temporary directory to store the executable.
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    poplar_exec_filepath = os.path.join(tmp_dir, "application.poplar_exec")

    # Compile the application.
    compile_op = application_compile_op.experimental_application_compile_op(
        my_net, output_path=poplar_exec_filepath)

    with sl.Session() as sess:
      sess.run(compile_op)

    # Create the start op.
    # This creates the poplar engine in a background thread.
    engine_name = "my_engine"
    ctx = embedded_runtime.embedded_runtime_start(poplar_exec_filepath, [],
                                                  engine_name)

    # Create the call op and the input placeholder.
    input_placeholder_0 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_1 = array_ops.placeholder(np.float32, shape=[1, 5])
    input_placeholder_2 = array_ops.placeholder(np.float32, shape=[5, 1])
    input_placeholder_3 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_4 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_5 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_6 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_7 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_8 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_9 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_10 = array_ops.placeholder(np.float32, shape=[])
    input_placeholder_11 = array_ops.placeholder(np.float32, shape=[])

    call_result = embedded_runtime.embedded_runtime_call([
        input_placeholder_0, input_placeholder_1, input_placeholder_2,
        input_placeholder_3, input_placeholder_4, input_placeholder_5,
        input_placeholder_6, input_placeholder_7, input_placeholder_8,
        input_placeholder_9, input_placeholder_10, input_placeholder_11
    ], ctx)

    for _ in range(loop_count):
      with sl.Session() as sess:
        # Expect execution without throwing an exception.
        sess.run(call_result,
                 feed_dict={
                     input_placeholder_0: np.ones(()).astype(np.float32),
                     input_placeholder_1: np.ones((1, 5)).astype(np.float32),
                     input_placeholder_2: np.ones((5, 1)).astype(np.float32),
                     input_placeholder_3: np.ones(()).astype(np.float32),
                     input_placeholder_4: np.ones(()).astype(np.float32),
                     input_placeholder_5: np.ones(()).astype(np.float32),
                     input_placeholder_6: np.ones(()).astype(np.float32),
                     input_placeholder_7: np.ones(()).astype(np.float32),
                     input_placeholder_8: np.ones(()).astype(np.float32),
                     input_placeholder_9: np.ones(()).astype(np.float32),
                     input_placeholder_10: np.ones(()).astype(np.float32),
                     input_placeholder_11: np.ones(()).astype(np.float32),
                 })


if __name__ == "__main__":
  googletest.main()

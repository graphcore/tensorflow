# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("@local_config_ipu_horovod//:build_defs_horovod.bzl", "if_horovod", "poprun_py_test")

py_library(
    name = "ipu_ops_lib",
    srcs = [
        "ops/cross_replica_ops.py",
        "ops/embedding_ops.py",
        "ops/functional_ops.py",
        "ops/image_ops.py",
        "ops/internal_ops.py",
        "ops/math_ops.py",
        "ops/nn_ops.py",
        "ops/normalization_ops.py",
        "ops/op_util.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/rnn_ops.py",
        "ops/slicing_ops.py",
        "ops/statistics_ops.py",
        "ops/summary_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:dataset_benchmark_py",
        "//tensorflow/compiler/plugin/poplar:functional_ops_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar:popfloat_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

py_library(
    name = "ipu_grad_ops_lib",
    srcs = [
        "ops/all_to_all_op_grad.py",
        "ops/cross_replica_ops_grad.py",
        "ops/embedding_ops_grad.py",
        "ops/experimental/popfloat_ops_grad.py",
        "ops/functional_ops_grad.py",
        "ops/internal_ops_grad.py",
        "ops/nn_ops_grad.py",
        "ops/normalization_ops_grad.py",
        "ops/pipelining_ops_grad.py",
        "ops/rand_ops_grad.py",
        "ops/rnn_ops_grad.py",
        "ops/slicing_ops_grad.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:functional_ops_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
    ],
)

py_library(
    name = "data",
    srcs = [
        "data/__init__.py",
        "data/ops/__init__.py",
        "data/ops/dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/compiler/plugin/poplar:dataset_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python/distribute:values",
    ],
)

py_library(
    name = "dataset_extractor",
    srcs = [
        "dataset_extractor.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:dataset_exporters_py",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "ipu_lib",
    srcs = [
        "__init__.py",
        "config.py",
        "dataset_benchmark.py",
        "ipu_compiler.py",
        "ipu_estimator.py",
        "ipu_infeed_queue.py",
        "ipu_multi_worker_strategy.py",
        "ipu_outfeed_queue.py",
        "ipu_pipeline_estimator.py",
        "ipu_run_config.py",
        "ipu_session_run_hooks.py",
        "ipu_strategy.py",
        "keras/__init__.py",
        "keras/extensions/__init__.py",
        "keras/extensions/data_adapter.py",
        "keras/extensions/functional_extensions.py",
        "keras/extensions/model_extensions.py",
        "keras/extensions/sequential_extensions.py",
        "keras/layer_replacement.py",
        "keras/layers/__init__.py",
        "keras/layers/ctc.py",
        "keras/layers/dropout.py",
        "keras/layers/embedding_lookup.py",
        "keras/layers/ipu_layer.py",
        "keras/layers/math.py",
        "keras/layers/normalization.py",
        "keras/layers/recomputation.py",
        "keras/layers/rnn.py",
        "keras/losses/__init__.py",
        "keras/losses/ctc.py",
        "keras/model.py",
        "keras/optimizers/__init__.py",
        "keras/optimizers/cross_replica_optimizer.py",
        "keras/optimizers/gradient_accumulate_optimizer.py",
        "keras/optimizers/ipu_wrappers.py",
        "keras/optimizers/map_gradient_optimizer.py",
        "keras/pipeline.py",
        "loops.py",
        "ops/all_to_all_op.py",
        "ops/cross_replica_ops.py",
        "ops/custom_ops.py",
        "ops/embedding_ops.py",
        "ops/experimental/popfloat_cast_to_gfloat.py",
        "ops/functional_ops.py",
        "ops/image_ops.py",
        "ops/internal_ops.py",
        "ops/math_ops.py",
        "ops/nn_ops.py",
        "ops/normalization_ops.py",
        "ops/op_util.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/reduce_scatter_op.py",
        "ops/replication_ops.py",
        "ops/rnn_ops.py",
        "ops/slicing_ops.py",
        "ops/statistics_ops.py",
        "ops/summary_ops.py",
        "optimizers/__init__.py",
        "optimizers/cross_replica_optimizer.py",
        "optimizers/gradient_accumulation_optimizer.py",
        "optimizers/ipu_optimizer.py",
        "optimizers/map_gradient_optimizer.py",
        "optimizers/sharded_optimizer.py",
        "scopes.py",
        "sharding.py",
        "sharding_utils.py",
        "utils.py",
        "vertex_edsl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data",
        ":dataset_extractor",
        ":ipu_grad_ops_lib",
        ":ipu_ops_lib",
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar:sendrecv_ops_py",
        "//tensorflow/compiler/plugin/poplar/tools:tensorflow_weights_extractor_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/estimator:estimator_py",
    ] + if_horovod(["//tensorflow/python/ipu/horovod:horovod_lib"]),
)

tf_py_test(
    name = "config_test",
    size = "small",
    srcs = ["tests/config_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "reconfigure_ipu_test",
    size = "small",
    srcs = ["tests/reconfigure_ipu_test.py"],
    shard_count = 7,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "gcl_test",
    size = "medium",
    srcs = ["tests/gcl_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "io_tiles_test",
    size = "medium",
    srcs = ["tests/io_tiles_test.py"],
    shard_count = 6,  # as can't have differing config in same process
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "report_tf2_test",
    size = "medium",
    srcs = ["tests/report_tf2_test.py"],
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "map_gradient_optimizer_test",
    size = "medium",
    srcs = ["tests/map_gradient_optimizer_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "expression_op_test",
    size = "medium",
    srcs = ["tests/expression_op_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "dropout_test",
    size = "medium",
    srcs = ["tests/dropout_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "nn_ops_test",
    size = "medium",
    srcs = ["tests/nn_ops_test.py"],
    shard_count = 8,
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "embedding_lookup_test",
    size = "large",
    srcs = ["tests/embedding_lookup_test.py"],
    shard_count = 12,
    tags = ["hw_poplar_test"],
    deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "keras_embedding_lookup_test",
    size = "large",
    srcs = ["tests/keras/keras_embedding_lookup_test.py"],
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_rnn_test",
    size = "large",
    srcs = ["tests/keras/keras_rnn_test.py"],
    shard_count = 5,
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_pipeline_test",
    size = "large",
    srcs = ["tests/keras/keras_pipeline_test.py"],
    shard_count = 27,
    tags = ["isolated_poplar_tests"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_pipeline_recomputation_test",
    size = "large",
    srcs = ["tests/keras/keras_pipeline_recomputation_test.py"],
    shard_count = 2,
    tags = ["isolated_poplar_tests"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "keras_pipeline_input_validation_test",
    size = "medium",
    srcs = ["tests/keras/keras_pipeline_input_validation_test.py"],
    tags = ["isolated_poplar_tests"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_sequential_pipeline_test",
    size = "large",
    srcs = ["tests/keras/keras_sequential_pipeline_test.py"],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_model_test",
    size = "large",
    srcs = ["tests/keras/keras_model_test.py"],
    shard_count = 12,
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_functional_model_test",
    size = "large",
    srcs = ["tests/keras/keras_functional_model_test.py"],
    shard_count = 12,
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_pipeline_save_restore_test",
    size = "large",
    srcs = ["tests/keras/keras_pipeline_save_restore_test.py"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_sequential_pipeline_save_restore_test",
    size = "large",
    srcs = ["tests/keras/keras_sequential_pipeline_save_restore_test.py"],
    tags = ["isolated_poplar_tests"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_layer_replacement_test",
    size = "large",
    srcs = ["tests/keras/keras_layer_replacement_test.py"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_normalization_test",
    size = "large",
    srcs = ["tests/keras/keras_normalization_test.py"],
    shard_count = 12,
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_ctc_test",
    size = "large",
    srcs = ["tests/keras/keras_ctc_test.py"],
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_dropout_test",
    size = "medium",
    srcs = ["tests/keras/keras_dropout_test.py"],
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_mnist_test",
    size = "large",
    srcs = ["tests/keras/examples/keras_mnist_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "keras_synthetic_data_test",
    size = "large",
    srcs = ["tests/keras/keras_synthetic_data_test.py"],
    # Shard count set equal to the number of tests as one process per test is
    # required
    tags = ["hw_poplar_test"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_v2_optimizers",
    size = "large",
    srcs = ["tests/keras/keras_v2_optimizers.py"],
    shard_count = 5,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "math_ops_test",
    size = "large",
    srcs = ["tests/math_ops_test.py"],
    shard_count = 40,
    deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "math_layers_test",
    size = "large",
    srcs = ["tests/math_layers_test.py"],
    shard_count = 12,
    deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["tests/estimator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "fifo_test",
    size = "large",
    srcs = ["tests/fifo_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "iteration_counter_test",
    size = "large",
    srcs = ["tests/iteration_counter_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "host_embedding_lookup_test",
    size = "large",
    srcs = ["tests/host_embedding_lookup_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "host_embedding_lookup_ga_test",
    size = "large",
    srcs = ["tests/host_embedding_lookup_ga_test.py"],
    deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "random_constant_matmul_test",
    size = "large",
    srcs = ["tests/random_constant_matmul_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "recompute_suggestion_test",
    size = "large",
    srcs = ["tests/recompute_suggestion_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "infeed_outfeed_test",
    size = "large",
    srcs = ["tests/infeed_outfeed_test.py"],
    shard_count = 10,
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "infeed_outfeed_overlap_test",
    size = "large",
    srcs = ["tests/infeed_outfeed_overlap_test.py"],
    shard_count = 10,
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "outfeed_ordinal_test",
    size = "medium",
    srcs = ["tests/outfeed_ordinal_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "dataset_benchmark_test",
    size = "medium",
    srcs = ["tests/dataset_benchmark_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_compiler_test",
    size = "large",
    srcs = ["tests/ipu_compiler_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_estimator_test",
    size = "large",
    srcs = ["tests/ipu_estimator_test.py"],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_estimator_replicated_test",
    size = "large",
    srcs = ["tests/ipu_estimator_replicated_test.py"],
    shard_count = 8,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_pipeline_estimator_test",
    size = "large",
    srcs = ["tests/ipu_pipeline_estimator_test.py"],
    shard_count = 8,
    tags = ["isolated_poplar_tests"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_multi_worker_strategy_test",
    size = "large",
    srcs = ["tests/ipu_multi_worker_strategy_test.py"],
    shard_count = 4,
    tags = [
        "hw_poplar_test",
        "isolated_poplar_tests",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_session_run_hooks_test",
    size = "large",
    srcs = ["tests/ipu_session_run_hooks_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_strategy_test",
    size = "large",
    srcs = ["tests/ipu_strategy_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_strategy_replicated_test",
    size = "large",
    srcs = ["tests/ipu_strategy_replicated_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "pop_datastream_test",
    size = "large",
    srcs = ["tests/pop_datastream_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "frontend_attributes_test",
    size = "medium",
    srcs = ["tests/frontend_attributes_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "popnn_normalization_test",
    size = "medium",
    srcs = ["tests/popnn_normalization_test.py"],
    shard_count = 6,
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "popops_cross_replica_sum_test",
    size = "medium",
    srcs = ["tests/popops_cross_replica_sum_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "replication_normalise_test",
    size = "medium",
    srcs = ["tests/replication_normalise_test.py"],
    srcs_version = "PY2AND3",
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras",
    ],
)

cc_binary(
    name = "libadd_incrementing_custom.so",
    srcs = ["tests/add_incrementing_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libadd_tensors_custom.so",
    srcs = ["tests/add_tensors_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libadd_partial_gradients_custom.so",
    srcs = ["tests/add_partial_gradients_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libadd_incrementing_custom_with_metadata.so",
    srcs = ["tests/add_incrementing_custom_with_metadata.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

cc_binary(
    name = "libwrong_api_level_custom.so",
    srcs = ["tests/wrong_api_level_custom.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_py_test(
    name = "user_ops_test",
    size = "large",
    srcs = ["tests/user_ops_test.py"],
    data = [
        "tests/add_scaled_vector_add_codelet.cc",
        ":libadd_incrementing_custom.so",
        ":libadd_incrementing_custom_with_metadata.so",
        ":libadd_partial_gradients_custom.so",
        ":libadd_tensors_custom.so",
        ":libwrong_api_level_custom.so",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "utils_test",
    size = "medium",
    srcs = ["tests/utils_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "utils_test_v2",
    size = "medium",
    srcs = ["tests/utils_test_v2.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "utils_use_synthetic_data_for_test",
    size = "medium",
    srcs = ["tests/utils_use_synthetic_data_for_test.py"],
    # Shard count set equal to the number of tests as one process per test is
    # required
    shard_count = 7,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
    ],
)

tf_py_test(
    name = "utils_get_config_test",
    size = "medium",
    srcs = ["tests/utils_get_config_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_library(
    name = "pipelining_test_util",
    testonly = 1,
    srcs = ["tests/pipelining_test_util.py"],
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_test",
    size = "large",
    srcs = ["tests/pipelining_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 37,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_grouped_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_recomputation_test.py"],
    data = [
        ":libpipelining_stateful_op.so",
    ],
    shard_count = 25,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_grouped_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_test.py"],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_grouped_overlap_test",
    size = "large",
    srcs = ["tests/pipelining_grouped_overlap_test.py"],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

cc_binary(
    name = "libpipelining_stateful_op.so",
    srcs = ["tests/pipelining_stateful_op.cc"],
    copts = ["-fexceptions"],
    linkshared = True,
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
)

tf_py_test(
    name = "pipelining_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_recomputation_test.py"],
    data = [
        ":libpipelining_stateful_op.so",
    ],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_sequential_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_sequential_recomputation_test.py"],
    shard_count = 16,
    tags = ["hw_poplar_test"],
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_sequential_test",
    size = "large",
    srcs = ["tests/pipelining_sequential_test.py"],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_recomputation_test.py"],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_test.py"],
    shard_count = 16,
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_conv_classify_test",
    size = "large",
    srcs = ["tests/pipelining_conv_classify_test.py"],
    shard_count = 4,
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pipelining_recomputation_conv_classify_test",
    size = "large",
    srcs = ["tests/pipelining_recomputation_conv_classify_test.py"],
    shard_count = 2,
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dataset_ops_test",
    size = "medium",
    srcs = ["tests/dataset_ops_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "functional_ops_test",
    size = "large",
    srcs = ["tests/functional_ops_test.py"],
    shard_count = 4,
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "gradient_accumulation_test",
    size = "large",
    srcs = ["tests/gradient_accumulation_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 11,
    tags = ["hw_poplar_test"],
    deps = [
        ":ipu_lib",
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["tests/image_ops_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "multi_conv_test",
    size = "large",
    srcs = ["tests/multi_conv_test.py"],
    shard_count = 5,
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "group_conv_test",
    size = "large",
    srcs = ["tests/group_conv_test.py"],
    deps = [
        ":ipu_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "candidate_sampler_test",
    size = "large",
    srcs = ["tests/candidate_sampler_test.py"],
    shard_count = 7,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "seed_control_test",
    size = "large",
    srcs = ["tests/seed_control_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "reductions_test",
    size = "large",
    srcs = ["tests/reductions_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "keras_rnn_hw_test",
    size = "large",
    srcs = ["tests/keras_rnn_hw_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "random_number_generation_test",
    size = "large",
    srcs = ["tests/random_number_generation_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replicated_graph_test",
    size = "large",
    srcs = ["tests/replicated_graph_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 12,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replicated_index_test",
    size = "large",
    srcs = ["tests/replicated_index_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replicated_reduce_scatter_test",
    size = "large",
    srcs = ["tests/replicated_reduce_scatter_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replicated_seed_control_test",
    size = "large",
    srcs = ["tests/replicated_seed_control_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replicated_stateful_gradient_accumulate_test",
    size = "large",
    srcs = ["tests/replicated_stateful_gradient_accumulate_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replication_all_to_all_test",
    size = "large",
    srcs = ["tests/replication_all_to_all_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "replication_grouped_collectives_test",
    size = "large",
    srcs = ["tests/replication_grouped_collectives_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "sharded_and_replicated_test",
    size = "large",
    srcs = ["tests/sharded_and_replicated_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 5,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "floating_point_control_bits_test",
    size = "large",
    srcs = ["tests/floating_point_control_bits_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 11,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "dropout_hw_test",
    size = "large",
    srcs = ["tests/dropout_hw_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 7,
    tags = ["hw_poplar_test"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "pipelining_batch_serial_sequential_hw_test",
    size = "large",
    srcs = ["tests/pipelining_batch_serial_sequential_hw_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 4,
    tags = ["hw_poplar_test"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "pipelining_hw_test",
    size = "large",
    srcs = ["tests/pipelining_hw_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 10,
    tags = ["hw_poplar_test"],
    deps = [
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

poprun_py_test(
    name = "poprun_replica_partitioning_test",
    size = "large",
    srcs = ["tests/poprun_replica_partitioning_test.py"],
    ipus_per_replica = 2,
    main = "tests/poprun_replica_partitioning_test.py",
    num_instances = 2,
    num_replicas = 8,
    deps = [
        ":pipelining_test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "keras_execution_parameters_test",
    size = "large",
    srcs = ["tests/keras/keras_execution_parameters_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 13,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_model_replication_mnist_test",
    size = "large",
    srcs = ["tests/keras/keras_model_replication_mnist_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_model_replication_test",
    size = "large",
    srcs = ["tests/keras/keras_model_replication_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_models_comparison_test",
    size = "large",
    srcs = ["tests/keras/keras_models_comparison_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_pipeline_model_replication_test",
    size = "large",
    srcs = ["tests/keras/keras_pipeline_model_replication_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_pipeline_sequential_replication_test",
    size = "large",
    srcs = ["tests/keras/keras_pipeline_sequential_replication_test.py"],
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_sequential_replication_test",
    size = "large",
    srcs = ["tests/keras/keras_sequential_replication_test.py"],
    # Shard count needs to match the number of tests.
    shard_count = 2,
    tags = ["hw_poplar_test"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sequence_slice_test",
    size = "large",
    srcs = ["tests/sequence_slice_test.py"],
    shard_count = 12,
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "histogram_test",
    size = "medium",
    srcs = ["tests/histogram_test.py"],
    shard_count = 8,
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "candidate_sampler_test",
        "config_test",
        "dataset_benchmark_test",
        "dataset_ops_test",
        "dropout_hw_test",
        "dropout_test",
        "embedding_lookup_test",
        "estimator_test",
        "expression_op_test",
        "fifo_test",
        "floating_point_control_bits_test",
        "frontend_attributes_test",
        "functional_ops_test",
        "gcl_test",
        "gradient_accumulation_test",
        "group_conv_test",
        "host_embedding_lookup_ga_test",
        "host_embedding_lookup_test",
        "image_ops_test",
        "infeed_outfeed_overlap_test",
        "infeed_outfeed_test",
        "io_tiles_test",
        "ipu_compiler_test",
        "ipu_estimator_replicated_test",
        "ipu_estimator_test",
        "ipu_multi_worker_strategy_test",
        "ipu_pipeline_estimator_test",
        "ipu_session_run_hooks_test",
        "ipu_strategy_replicated_test",
        "ipu_strategy_test",
        "iteration_counter_test",
        "keras_ctc_test",
        "keras_dropout_test",
        "keras_embedding_lookup_test",
        "keras_execution_parameters_test",
        "keras_functional_model_test",
        "keras_model_replication_mnist_test",
        "keras_model_replication_test",
        "keras_model_test",
        "keras_models_comparison_test",
        "keras_normalization_test",
        "keras_pipeline_model_replication_test",
        "keras_pipeline_recomputation_test",
        "keras_pipeline_save_restore_test",
        "keras_pipeline_sequential_replication_test",
        "keras_pipeline_test",
        "keras_rnn_hw_test",
        "keras_rnn_test",
        "keras_sequential_pipeline_save_restore_test",
        "keras_sequential_pipeline_test",
        "keras_sequential_replication_test",
        "keras_synthetic_data_test",
        "keras_v2_optimizers",
        "map_gradient_optimizer_test",
        "math_layers_test",
        "math_ops_test",
        "multi_conv_test",
        "nn_ops_test",
        "outfeed_ordinal_test",
        "pipelining_batch_serial_sequential_hw_test",
        "pipelining_batch_serial_sequential_recomputation_test",
        "pipelining_batch_serial_sequential_test",
        "pipelining_conv_classify_test",
        "pipelining_grouped_overlap_test",
        "pipelining_grouped_recomputation_test",
        "pipelining_grouped_test",
        "pipelining_hw_test",
        "pipelining_recomputation_conv_classify_test",
        "pipelining_recomputation_test",
        "pipelining_sequential_recomputation_test",
        "pipelining_sequential_test",
        "pipelining_test",
        "pop_datastream_test",
        "popnn_normalization_test",
        "popops_cross_replica_sum_test",
        "poprun_replica_partitioning_test",
        "random_constant_matmul_test",
        "random_number_generation_test",
        "recompute_suggestion_test",
        "reconfigure_ipu_test",
        "reductions_test",
        "replicated_graph_test",
        "replicated_index_test",
        "replicated_reduce_scatter_test",
        "replicated_seed_control_test",
        "replicated_stateful_gradient_accumulate_test",
        "replication_all_to_all_test",
        "replication_grouped_collectives_test",
        "replication_normalise_test",
        "seed_control_test",
        "sequence_slice_test",
        "sharded_and_replicated_test",
        "user_ops_test",
        "utils_get_config_test",
        "utils_test",
        "utils_test_v2",
        "utils_use_synthetic_data_for_test",
        "//tensorflow/python/ipu/tests/keras/extensions:extensions_test_suite",
    ],
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
        ":all_tests",
        "//tensorflow/compiler/plugin/poplar:poplar_ci_test_suite",
    ] + if_horovod(["//tensorflow/python/ipu/horovod:horovod_test_suite"]),
)

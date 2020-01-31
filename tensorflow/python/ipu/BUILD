# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")

# Needed so we can build the custom op.
load("@local_config_poplar//poplar:build_defs.bzl", "tf_poplar_build_tag")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

py_library(
    name = "ipu_ops_lib",
    srcs_version = "PY2AND3",
    srcs = [
        "ops/cross_replica_ops.py",
        "ops/embedding_ops.py",
        "ops/internal_ops.py",
        "ops/normalization_ops.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/rnn_ops.py",
        "ops/summary_ops.py",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:dataset_benchmark_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:pipelining_ops_py",
        "//tensorflow/compiler/plugin/poplar:popfloat_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

py_library(
    name = "ipu_grad_ops_lib",
    srcs_version = "PY2AND3",
    srcs = [
        "ops/all_to_all_op_grad.py",
        "ops/embedding_ops_grad.py",
        "ops/experimental/popfloat_ops_grad.py",
        "ops/internal_ops_grad.py",
        "ops/pipelining_ops_grad.py",
        "ops/normalization_ops_grad.py",
        "ops/rand_ops_grad.py",
        "ops/rnn_ops_grad.py",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:pipelining_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
    ],
)

py_library(
    name = "utils",
    srcs_version = "PY2AND3",
    srcs = [
        "utils.py"
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/python/distribute:values",
    ],
)

py_library(
    name = "data",
    srcs_version = "PY2AND3",
    srcs = [
        "data/__init__.py",
        "data/ops/__init__.py",
        "data/ops/dataset_ops.py"
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:dataset_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/python/distribute:values",
    ],
)

py_library(
    name = "ipu_lib",
    srcs_version = "PY2AND3",
    srcs = [
        "__init__.py",
        "keras/__init__.py",
        "keras/layers/__init__.py",
        "autoshard.py",
        "autoshard_cnn.py",
        "dataset_benchmark.py",
        "gradient_accumulation_optimizer.py",
        "keras/sequential.py",
        "keras/layers/RNN.py",
        "ipu_compiler.py",
        "ipu_estimator.py",
        "ipu_infeed_queue.py",
        "ipu_multi_worker_strategy.py",
        "ipu_optimizer.py",
        "ipu_outfeed_queue.py",
        "ipu_pipeline_estimator.py",
        "ipu_run_config.py",
        "ipu_strategy.py",
        "loops.py",
        "ops/all_to_all_op.py",
        "ops/cross_replica_ops.py",
        "ops/custom_ops.py",
        "ops/embedding_ops.py",
        "ops/experimental/popfloat_cast_to_gfloat.py",
        "ops/internal_ops.py",
        "ops/normalization_ops.py",
        "ops/pipelining_ops.py",
        "ops/rand_ops.py",
        "ops/reduce_scatter_op.py",
        "ops/replication_ops.py",
        "ops/rnn_ops.py",
        "ops/summary_ops.py",
        "optimizers/cross_replica_optimizer.py",
        "optimizers/gradient_accumulation_optimizer.py",
        "optimizers/map_gradient_optimizer.py",
        "optimizers/sharded_optimizer.py",
        "scopes.py",
        "sharded_optimizer.py",
        "sharding.py",
        "sharding_utils.py",
        "vertex_edsl.py",
    ],
    deps = [
        ":data",
        ":utils",
        ":ipu_ops_lib",
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/plugin/poplar:sendrecv_ops_py",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

tf_py_test(
    name = "autoshard_cnn_test",
    size = "medium",
    srcs = ["tests/autoshard_cnn_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "map_gradient_optimizer_test",
    size = "medium",
    srcs = ["tests/map_gradient_optimizer_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "expression_op_test",
    size = "medium",
    srcs = ["tests/expression_op_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "dropout_test",
    size = "small",
    srcs = ["tests/dropout_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "embedding_lookup_test",
    size = "large",
    srcs = ["tests/embedding_lookup_test.py"],
    additional_deps = [
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "RNN_test",
    size = "medium",
    srcs = ["tests/keras/RNN_test.py"],
    additional_deps = [
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "medium",
    srcs = ["tests/keras/sequential_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["tests/estimator_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
    srcs_version = "PY2AND3",
)

tf_py_test(
    name = "fifo_test",
    size = "small",
    srcs = ["tests/fifo_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)


tf_py_test(
    name = "random_constant_matmul_test",
    size = "small",
    srcs = ["tests/random_constant_matmul_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "recompute_suggestion_test",
    size = "small",
    srcs = ["tests/recompute_suggestion_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "infeed_outfeed_test",
    size = "large",
    shard_count = 2,
    srcs = ["tests/infeed_outfeed_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dataset_benchmark_test",
    size = "medium",
    srcs = ["tests/dataset_benchmark_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ipu_compiler_test",
    size = "small",
    srcs = ["tests/ipu_compiler_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_estimator_test",
    size = "large",
    tags = ['exclusive'],
    shard_count = 8,
    srcs = ["tests/ipu_estimator_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ipu_estimator_replicated_test",
    size = "large",
    shard_count = 2,
    srcs = ["tests/ipu_estimator_replicated_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ipu_pipeline_estimator_test",
    tags = ['exclusive'],
    size = "large",
    shard_count = 8,
    srcs = ["tests/ipu_pipeline_estimator_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ipu_multi_worker_strategy_test",
    tags = ['exclusive'],
    size = "large",
    shard_count = 4,
    srcs = ["tests/ipu_multi_worker_strategy_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

tf_py_test(
    name = "ipu_strategy_test",
    size = "large",
    srcs = ["tests/ipu_strategy_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
    ],
)

py_test(
    name = "pop_datastream_test",
    size = "small",
    srcs = ["tests/pop_datastream_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
)

tf_py_test(
    name = "frontend_attributes_test",
    size = "small",
    srcs = ["tests/frontend_attributes_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)

tf_py_test(
    name = "popnn_normalization_test",
    size = "small",
    srcs = ["tests/popnn_normalization_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "popops_cross_replica_sum_test",
    size = "medium",
    srcs = ["tests/popops_cross_replica_sum_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "replication_normalise_test",
    size = "medium",
    srcs = ["tests/replication_normalise_test.py"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras:layers",
    ],
    srcs_version = "PY2AND3",
)

cc_binary(
    name = "libadd_incrementing_custom.so",
    srcs = ["tests/add_incrementing_custom.cc"],
    copts = ["-fexceptions"],
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
    linkshared = True,
)

cc_binary(
    name = "libadd_incrementing_custom_with_metadata.so",
    srcs = ["tests/add_incrementing_custom_with_metadata.cc"],
    copts = ["-fexceptions"],
    deps = [
        "@local_config_poplar//poplar:poplar_libs",
    ],
    linkshared = True,
)
tf_py_test(
    name = "user_ops_test",
    size = "small",
    srcs = ["tests/user_ops_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    data = [
        ":libadd_incrementing_custom.so",
        ":libadd_incrementing_custom_with_metadata.so",
    ],
)

tf_py_test(
    name = "utils_test",
    size = "small",
    srcs = ["tests/utils_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

py_library(
    name = "pipelining_test_util",
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
    shard_count = 16,
    additional_deps = [
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
    shard_count = 4,
    additional_deps = [
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
    additional_deps = [
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
    name = "pipelining_recomputation_test",
    size = "large",
    srcs = ["tests/pipelining_recomputation_test.py"],
    shard_count = 4,
    additional_deps = [
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
    additional_deps = [
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
    additional_deps = [
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
    additional_deps = [
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
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "autoshard_cnn_test",
        "dataset_benchmark_test",
        "dataset_ops_test",
        "dropout_test",
        "embedding_lookup_test",
        "expression_op_test",
        "estimator_test",
        "fifo_test",
        "frontend_attributes_test",
        "infeed_outfeed_test",
        "ipu_compiler_test",
        "ipu_estimator_test",
        "ipu_multi_worker_strategy_test",
        "ipu_pipeline_estimator_test",
        "ipu_strategy_test",
        "map_gradient_optimizer_test",
        "pipelining_conv_classify_test",
        "pipelining_grouped_recomputation_test",
        "pipelining_grouped_test",
        "pipelining_recomputation_test",
        "pipelining_sequential_recomputation_test",
        "pipelining_sequential_test",
        "pipelining_test",
        "pop_datastream_test",
        "popnn_normalization_test",
        "popops_cross_replica_sum_test",
        "random_constant_matmul_test",
        "sequential_test",
        "RNN_test",
        "recompute_suggestion_test",
        "user_ops_test",
        "utils_test",
    ],
)

test_suite(
    name = "replicated_tests",
    tests = [
        "replication_normalise_test",
        "ipu_estimator_replicated_test"
    ],
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
      "//tensorflow/compiler/plugin/poplar:poplar_ci_test_suite",
      "//tensorflow/python/ipu:all_tests",
    ],
)

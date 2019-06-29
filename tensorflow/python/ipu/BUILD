# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")


py_library(
    name = "ipu_ops_lib",
    srcs_version = "PY2AND3",
    srcs = [
        "ops/cross_replica_ops.py",
        "ops/embedding_ops.py",
        "ops/internal_ops.py",
        "ops/normalization_ops.py",
        "ops/rand_ops.py",
        "ops/rnn_ops.py",
        "ops/summary_ops.py",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
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
        "ops/internal_ops_grad.py",
        "ops/normalization_ops_grad.py",
        "ops/rand_ops_grad.py",
        "ops/rnn_ops_grad.py",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/plugin/poplar:popops_ops_py",
        "//tensorflow/compiler/plugin/poplar:poprand_ops_py",
        "//tensorflow/compiler/plugin/poplar:poputil_ops_py",
    ],
)

py_library(
    name = "ipu_lib",
    srcs_version = "PY2AND3",
    srcs = [
        "__init__.py",
        "autoshard.py",
        "autoshard_cnn.py",
        "gradient_accumulation_optimizer.py",
        "ipu_compiler.py",
        "ipu_estimator.py",
        "ipu_infeed_queue.py",
        "ipu_optimizer.py",
        "ipu_outfeed_queue.py",
        "ipu_run_config.py",
        "loops.py",
        "ops/cross_replica_ops.py",
        "ops/embedding_ops.py",
        "ops/internal_ops.py",
        "ops/normalization_ops.py",
        "ops/rand_ops.py",
        "ops/rnn_ops.py",
        "ops/summary_ops.py",
        "scopes.py",
        "sharded_optimizer.py",
        "sharding.py",
        "sharding_utils.py",
        "utils.py",
    ],
    deps = [
        ":ipu_ops_lib",
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
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
    size = "medium",
    srcs = ["tests/embedding_lookup_test.py"],
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
    name = "ipu_estimator_test",
    size = "medium",
    srcs = ["tests/ipu_estimator_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
    srcs_version = "PY2AND3",
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

tf_py_test(
    name = "utils_test",
    size = "small",
    srcs = ["tests/utils_test.py"],
    additional_deps = [
        "//tensorflow/compiler/plugin/poplar:test_utils_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/ipu:ipu_lib",
        "//tensorflow/python/keras:layers",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "autoshard_cnn_test",
        "dropout_test",
        "embedding_lookup_test",
        "estimator_test",
        "fifo_test",
        "infeed_outfeed_test",
        "ipu_estimator_test",
        "pop_datastream_test",
        "popnn_normalization_test",
        "popops_cross_replica_sum_test",
        "utils_test",
    ],
)

test_suite(
    name = "replicated_tests",
    tests = [
        "replication_normalise_test",
    ],
)

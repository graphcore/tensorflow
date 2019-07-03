# Description:
# Python support for IPU.
#

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

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
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
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
        "ipu_infeed_queue.py",
        "ipu_outfeed_queue.py",
        "ipu_optimizer.py",
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
        "utils.py",
    ],
    deps = [
        ":ipu_ops_lib",
        ":ipu_grad_ops_lib",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
    ],
)

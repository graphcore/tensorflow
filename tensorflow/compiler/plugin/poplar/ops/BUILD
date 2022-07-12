load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library", "tf_gen_op_wrapper_py")
load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "common_shape_fns",
    srcs = ["common_shape_fns.cc"],
    hdrs = ["common_shape_fns.h"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

# When creating a new set of ops the following needs to be done:
#
# 1. New poplar_cc_library:
# poplar_cc_library(
#     name = "new_ops",
#     srcs = ["new/new_ops.cc"],
#     deps = ["//tensorflow/core:framework"],
#     alwayslink = True,
# )
#
# 2. New python wrapper for those ops:
# tf_gen_op_wrapper_py(
#     name = "new_ops_wrapper_py",
#     out = "gen_new_ops.py",
#     deps = [":new_ops"],
#     visibility = ["//tensorflow:internal"],
# )
#
# 3. New custom op libary:
# tf_custom_op_py_library(
#     name = "new_ops_py",
#     kernels = [":new_ops"],
#     deps = [":new_ops_wrapper_py"],
#     srcs_version = "PY2AND3",
# )
#
# The cc library `new_ops` needs to be added as a dependency to:
# //tensorflow/compiler/plugin/poplar:poplar_lib
# The custom op py library `new_ops_py` needs to be added as a dependency to one
# of the relevant libs in //tensorflow/python/ipu.

poplar_cc_library(
    name = "sendrecv_ops",
    srcs = ["sendrecv_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

poplar_cc_library(
    name = "ipu_ops",
    srcs = ["ipu_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = True,
)

tf_gen_op_wrapper_py(
    name = "dataset_ops_wrapper_py",
    out = "gen_dataset_ops.py",
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/dataset:ops",
    ],
)

tf_custom_op_py_library(
    name = "dataset_ops_py",
    kernels = [":dataset_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":dataset_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "ipu_ops_wrapper_py",
    out = "gen_ipu_ops.py",
    deps = [":ipu_ops"],
)

tf_custom_op_py_library(
    name = "ipu_ops_py",
    kernels = [":ipu_ops"],
    visibility = ["//visibility:public"],
    deps = [":ipu_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "popnn_ops_wrapper_py",
    out = "gen_popnn_ops.py",
    deps = ["//tensorflow/compiler/plugin/poplar/ops/popnn:ops"],
)

tf_custom_op_py_library(
    name = "popnn_ops_py",
    kernels = [":popnn_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":popnn_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "popops_ops_wrapper_py",
    out = "gen_popops_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/compiler/plugin/poplar/ops/popops:ops"],
)

tf_custom_op_py_library(
    name = "popops_ops_py",
    kernels = [":popops_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":popops_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "popfloat_ops_wrapper_py",
    out = "gen_popfloat_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/compiler/plugin/poplar/ops/popfloat:ops"],
)

tf_custom_op_py_library(
    name = "popfloat_ops_py",
    kernels = [":popfloat_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":popfloat_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "poprand_ops_wrapper_py",
    out = "gen_poprand_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/compiler/plugin/poplar/ops/poprand:ops"],
)

tf_custom_op_py_library(
    name = "poprand_ops_py",
    kernels = [":poprand_ops"],
    visibility = ["//visibility:public"],
    deps = [":poprand_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "poputil_ops_wrapper_py",
    out = "gen_poputil_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/compiler/plugin/poplar/ops/poputil:ops"],
)

tf_custom_op_py_library(
    name = "poputil_ops_py",
    kernels = [":poputil_ops"],
    visibility = ["//visibility:public"],
    deps = [":poputil_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "pop_datastream_ops_wrapper_py",
    out = "gen_pop_datastream_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/datastream:ops",
    ],
)

tf_custom_op_py_library(
    name = "pop_datastream_ops_py",
    kernels = [":pop_datastream_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":pop_datastream_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "sendrecv_ops_wrapper_py",
    out = "gen_sendrecv_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [":sendrecv_ops"],
)

tf_custom_op_py_library(
    name = "sendrecv_ops_py",
    kernels = [":sendrecv_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":sendrecv_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "functional_ops_wrapper_py",
    out = "gen_functional_ops.py",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/functional:ops",
    ],
)

tf_custom_op_py_library(
    name = "functional_ops_py",
    kernels = [":functional_ops"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":functional_ops_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "dataset_benchmark_wrapper_py",
    out = "gen_dataset_benchmark.py",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/datastream:dataset_benchmark",
    ],
)

tf_custom_op_py_library(
    name = "dataset_benchmark_py",
    kernels = [":dataset_benchmark"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":dataset_benchmark_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "application_runtime_wrapper_py",
    out = "gen_application_runtime.py",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/application_runtime:ops",
    ],
)

tf_custom_op_py_library(
    name = "application_runtime_py",
    kernels = [":application_runtime"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":application_runtime_wrapper_py"],
)

tf_gen_op_wrapper_py(
    name = "dataset_exporters_wrapper_py",
    out = "gen_dataset_exporters.py",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops/datastream:exporters",
    ],
)

tf_custom_op_py_library(
    name = "dataset_exporters_py",
    kernels = [":dataset_exporters"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":dataset_exporters_wrapper_py"],
)

poplar_cc_library(
    name = "ops",
    deps = [
        ":ipu_ops",
        ":sendrecv_ops",
        "//tensorflow/compiler/plugin/poplar/ops/application_runtime",
        "//tensorflow/compiler/plugin/poplar/ops/application_runtime:application_compile",
        "//tensorflow/compiler/plugin/poplar/ops/dataset:dataset_ops",
        "//tensorflow/compiler/plugin/poplar/ops/datastream:dataset_benchmark",
        "//tensorflow/compiler/plugin/poplar/ops/datastream:exporters",
        "//tensorflow/compiler/plugin/poplar/ops/datastream:feeds",
        "//tensorflow/compiler/plugin/poplar/ops/datastream:host_embedding",
        "//tensorflow/compiler/plugin/poplar/ops/functional",
        "//tensorflow/compiler/plugin/poplar/ops/functional:pipelining",
        "//tensorflow/compiler/plugin/poplar/ops/popfloat:cast_to_gfloat",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:ctc_loss",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:gelu",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:gru",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:hard_sigmoid",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:lstm",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:norm",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:softmax",
        "//tensorflow/compiler/plugin/poplar/ops/popnn:swish",
        "//tensorflow/compiler/plugin/poplar/ops/popops:all_to_all",
        "//tensorflow/compiler/plugin/poplar/ops/popops:cross_replica_sum",
        "//tensorflow/compiler/plugin/poplar/ops/popops:f8_convert",
        "//tensorflow/compiler/plugin/poplar/ops/popops:histogram",
        "//tensorflow/compiler/plugin/poplar/ops/popops:multi_slice",
        "//tensorflow/compiler/plugin/poplar/ops/popops:normalise_image",
        "//tensorflow/compiler/plugin/poplar/ops/popops:reduce_scatter",
        "//tensorflow/compiler/plugin/poplar/ops/popops:sequence_slice",
        "//tensorflow/compiler/plugin/poplar/ops/popops:within_replicas",
        "//tensorflow/compiler/plugin/poplar/ops/poprand:dropout_tf",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:assume_equal_across_replicas",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:barrier",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:codelet_expression",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:device_sync",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:execution_counter",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:fifo",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:print_tensor",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:recompute",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:remap",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:remap_deduce",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:replication_factor",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:replication_index",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:stateful_gradient_accumulate",
        "//tensorflow/compiler/plugin/poplar/ops/poputil:user_operation",
    ],
    alwayslink = True,
)

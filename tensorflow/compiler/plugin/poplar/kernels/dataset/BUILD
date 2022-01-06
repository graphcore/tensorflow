#Description:
#   Kernels specific to the IPU.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_kernel_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

tf_kernel_library(
    name = "buffer_dataset_op",
    srcs = ["buffer_dataset_op.cc"],
    hdrs = ["buffer_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:dataset_ops",
        "//tensorflow/core/kernels/data:name_utils",
    ],
)

tf_cc_test(
    name = "buffer_dataset_op_test",
    size = "small",
    srcs = ["buffer_dataset_op_test.cc"],
    deps = [
        ":buffer_dataset_op",
        "//tensorflow/compiler/plugin/poplar/ops/dataset:ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/data:dataset_test_base",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//tensorflow/core/kernels/data:iterator_ops",
        "//tensorflow/core/kernels/data:tensor_slice_dataset_op",
    ],
)

tf_kernel_library(
    name = "dataset",
    deps = [
        ":buffer_dataset_op",
        "//tensorflow/compiler/plugin/poplar/ops/dataset:ops",
    ],
)

tf_kernel_library(
    name = "kernels",
    deps = [
        ":dataset",
    ],
)

test_suite(
    name = "dataset_tests",
    tests = [
        "buffer_dataset_op_test",
    ],
)

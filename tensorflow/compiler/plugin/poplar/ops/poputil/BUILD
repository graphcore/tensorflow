load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "assume_equal_across_replicas",
    srcs = [
        "assume_equal_across_replicas.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "barrier",
    srcs = [
        "barrier.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "codelet_expression",
    srcs = [
        "codelet_expression.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "device_sync",
    srcs = [
        "device_sync.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "execution_counter",
    srcs = [
        "execution_counter.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "fifo",
    srcs = [
        "fifo.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "print_tensor",
    srcs = [
        "print_tensor.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "recompute",
    srcs = [
        "recompute.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "remap_deduce",
    srcs = [
        "remap_deduce.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "remap",
    srcs = [
        "remap.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "replication_factor",
    srcs = [
        "replication_factor.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "replication_index",
    srcs = [
        "replication_index.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "stateful_gradient_accumulate",
    srcs = [
        "stateful_gradient_accumulate.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "user_operation",
    srcs = [
        "user_operation.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":assume_equal_across_replicas",
        ":barrier",
        ":codelet_expression",
        ":device_sync",
        ":execution_counter",
        ":fifo",
        ":print_tensor",
        ":recompute",
        ":remap",
        ":remap_deduce",
        ":replication_factor",
        ":replication_index",
        ":stateful_gradient_accumulate",
        ":user_operation",
    ],
)

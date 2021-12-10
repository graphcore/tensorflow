load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "assert",
    srcs = [
        "assert.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:assert",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "assume_equal_across_replicas",
    srcs = [
        "assume_equal_across_replicas.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:assume_equal_across_replicas",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "barrier",
    srcs = [
        "barrier.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:barrier",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "codelet_expression_op",
    srcs = [
        "codelet_expression_op.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:codelet_expression_op",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "copy_into",
    srcs = [
        "copy_into.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:copy_into",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "fifo",
    srcs = [
        "fifo.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:fifo",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "inter_tileset_copy",
    srcs = [
        "inter_tileset_copy.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_tileset_copy",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "inter_ipu_copy",
    srcs = [
        "inter_ipu_copy.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:inter_ipu_copy",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "print_tensor",
    srcs = [
        "print_tensor.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:print_tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "remap",
    srcs = [
        "remap.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remap",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "remap_deduce",
    srcs = [
        "remap_deduce.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remap_deduce",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "remote_parameter",
    srcs = [
        "remote_parameter.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:remote_parameter",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "replication_factor",
    srcs = [
        "replication_factor.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:replication_factor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "replication_index",
    srcs = [
        "replication_index.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "stateful_gradient_accumulate",
    srcs = [
        "stateful_gradient_accumulate.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_gradient_accumulate",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "stateful_noop",
    srcs = [
        "stateful_noop.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateful_noop",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "user_op",
    srcs = [
        "user_op.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:user_op_hlo",
    ],
    alwayslink = True,
)

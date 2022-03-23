load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "all_gather",
    srcs = [
        "all_gather.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "collective_reorder",
    srcs = [
        "collective_reorder.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:collective_reorder",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_scatter",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "conv_ops",
    srcs = [
        "conv_ops.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "elementwise",
    srcs = [
        "elementwise.cc",
    ],
    deps = [
        ":expression_helpers",
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:elementwise",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "expression_helpers",
    srcs = [
        "expression_helpers.cc",
    ],
    hdrs = [
        "expression_helpers.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "fusions",
    srcs = [
        "fusions.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:all_gather",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "gather_ops",
    srcs = [
        "gather_ops.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "histogram",
    srcs = [
        "histogram.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:histogram",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "multi_slice",
    srcs = [
        "multi_slice.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:multi_slice",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "normalise_image",
    srcs = [
        "normalise_image.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:normalise_image",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "random",
    srcs = [
        "random.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "reduce_many",
    srcs = [
        "reduce_many.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_many",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "reduce_scatter",
    srcs = [
        "reduce_scatter.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:reduce_scatter",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "scaled_inplace",
    srcs = [
        "scaled_inplace.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:scaled_inplace",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "scatter_ops",
    srcs = [
        "scatter_ops.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "select_scalars",
    srcs = [
        "select_scalars.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "sequence_slice",
    srcs = [
        "sequence_slice.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:sequence_slice",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "slice_apply",
    srcs = [
        "slice_apply.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:slice_apply",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "sort_ops",
    srcs = [
        "sort_ops.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "within_replicas",
    srcs = [
        "within_replicas.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:within_replicas",
    ],
    alwayslink = True,
)

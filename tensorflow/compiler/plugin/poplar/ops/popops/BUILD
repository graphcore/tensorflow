load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "all_to_all",
    srcs = [
        "all_to_all.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "cross_replica_sum",
    srcs = [
        "cross_replica_sum.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "f8_convert",
    srcs = [
        "f8_convert.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "fp8_ops",
    srcs = [
        "fp8_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "histogram",
    srcs = [
        "histogram.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "multi_slice",
    srcs = [
        "multi_slice.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "normalise_image",
    srcs = [
        "normalise_image.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "reduce_scatter",
    srcs = [
        "reduce_scatter.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "sequence_slice",
    srcs = [
        "sequence_slice.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "within_replicas",
    srcs = [
        "within_replicas.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":all_to_all",
        ":cross_replica_sum",
        ":f8_convert",
        ":fp8_ops",
        ":histogram",
        ":multi_slice",
        ":normalise_image",
        ":reduce_scatter",
        ":sequence_slice",
        ":within_replicas",
    ],
)

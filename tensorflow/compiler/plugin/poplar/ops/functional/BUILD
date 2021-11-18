licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

poplar_cc_library(
    name = "functional",
    srcs = [
        "functional.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/ops:common_shape_fns",
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "pipelining",
    srcs = [
        "pipelining.cc",
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
        ":functional",
        ":pipelining",
    ],
)

load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "all_gather",
    srcs = [
        "all_gather.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "broadcast",
    srcs = [
        "broadcast.cc",
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
        ":all_gather",
        ":all_reduce",
        ":broadcast",
    ],
)

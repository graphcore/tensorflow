licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

poplar_cc_library(
    name = "cast_to_gfloat",
    srcs = [
        "cast_to_gfloat.cc",
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
        ":cast_to_gfloat",
    ],
)

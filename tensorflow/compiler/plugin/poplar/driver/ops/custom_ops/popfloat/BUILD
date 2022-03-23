load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "cast_to_gfloat",
    srcs = [
        "cast_to_gfloat.cc",
    ],
    deps = [
        ":gfloat_ops_utils",
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "gfloat_ops_utils",
    srcs = [
        "gfloat_ops_utils.cc",
    ],
    hdrs = [
        "gfloat_ops_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

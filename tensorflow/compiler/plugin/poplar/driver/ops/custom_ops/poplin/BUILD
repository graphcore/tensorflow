load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "cholesky",
    srcs = [
        "cholesky.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "triangular_solve",
    srcs = [
        "triangular_solve.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "weights_transpose_chans_flip_xy",
    srcs = [
        "weights_transpose_chans_flip_xy.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:weights_transpose_chans_flip_xy",
    ],
    alwayslink = True,
)

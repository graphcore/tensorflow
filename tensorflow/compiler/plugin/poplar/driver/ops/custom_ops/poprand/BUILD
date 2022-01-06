load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

poplar_cc_library(
    name = "candidate_sampler",
    srcs = [
        "candidate_sampler.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools:poplar_range_sampler",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:candidate_sampler",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "dropout",
    srcs = [
        "dropout.cc",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:tensor",
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:dropout_hlo",
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
        "//tensorflow/compiler/plugin/poplar/driver/tools/custom_ops:stateless_random",
    ],
    alwayslink = True,
)

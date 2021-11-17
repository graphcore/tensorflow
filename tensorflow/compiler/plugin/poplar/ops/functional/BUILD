licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

cc_library(
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

cc_library(
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

cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":functional",
        ":pipelining",
    ],
)

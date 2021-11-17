licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

cc_library(
    name = "dataset_ops",
    srcs = [
        "dataset_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

cc_library(
    name = "ops",
    srcs = [],
    deps = [
        ":dataset_ops",
    ],
)

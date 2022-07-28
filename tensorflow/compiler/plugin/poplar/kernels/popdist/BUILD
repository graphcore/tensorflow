load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "poplar_cc_library")

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar:__subpackages__",
    "//tensorflow/python/ipu:__subpackages__",
])

poplar_cc_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:inplace_ops",
        "//third_party/eigen3",
        "@local_config_poplar//poplar:poplar_libs",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:inplace_ops",
        "//third_party/eigen3",
        "@local_config_poplar//poplar:poplar_libs",
    ],
    alwayslink = True,
)

poplar_cc_library(
    name = "popdist",
    deps = [
        ":all_reduce",
        ":broadcast",
    ],
)

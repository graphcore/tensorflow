licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

cc_library(
    name = "cast_to_gfloat",
    srcs = [
        "cast_to_gfloat.cc",
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
        ":cast_to_gfloat",
    ],
)

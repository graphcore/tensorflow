licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

cc_library(
    name = "dropout_tf",
    srcs = [
        "dropout_tf.cc",
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
        ":dropout_tf",
    ],
)

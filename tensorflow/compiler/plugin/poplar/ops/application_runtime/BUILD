licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

cc_library(
    name = "application_compile",
    srcs = [
        "application_compile.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = True,
)

cc_library(
    name = "application_runtime",
    srcs = [
        "application_runtime.cc",
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
        ":application_compile",
        ":application_runtime",
    ],
)

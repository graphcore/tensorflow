load("//tensorflow/compiler/plugin/poplar:poplar.bzl", "popit_cc_library")

licenses(["restricted"])

package(default_visibility = [
    "//tensorflow/compiler/plugin/poplar:__subpackages__",
])

popit_cc_library(
    name = "popit_platform_id",
    srcs = ["popit_platform_id.cc"],
    hdrs = ["popit_platform_id.h"],
    deps = [
        "//tensorflow/stream_executor:platform",
    ],
)

popit_cc_library(
    name = "popit_compiler",
    srcs = [
        "popit_compiler.cc",
    ],
    hdrs = [
        "popit_compiler.h",
    ],
    deps = [
        ":popit_platform_id",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tracepoint",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/cpu:cpu_executable",
    ],
)

popit_cc_library(
    name = "popit_backend",
    deps = [
        ":popit_compiler",
    ],
)

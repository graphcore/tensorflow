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
    name = "popit_executor",
    srcs = [
        "popit_executor.cc",
    ],
    hdrs = [
        "popit_executor.h",
        "popit_platform.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor",
        "//tensorflow/compiler/plugin/poplar/driver:xla_ipu_common",
        "//tensorflow/compiler/plugin/poplar/driver/tools:tracepoint",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/stream_executor:platform",
    ],
)

popit_cc_library(
    name = "popit_executable",
    srcs = [
        "popit_executable.cc",
    ],
    hdrs = [
        "popit_executable.h",
    ],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver/tools:tracepoint",
        "//tensorflow/compiler/xla/service:executable",
    ],
)

popit_cc_library(
    name = "popit_transfer_manager",
    srcs = [
        "popit_transfer_manager.cc",
    ],
    hdrs = [
        "popit_transfer_manager.h",
    ],
    deps = [
        ":popit_platform_id",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
    ],
)

popit_cc_library(
    name = "popit_platform",
    srcs = [
        "popit_platform.cc",
    ],
    hdrs = [
        "popit_platform.h",
    ],
    deps = [
        ":popit_executor",
        ":popit_platform_id",
        "//tensorflow/compiler/plugin/poplar/driver:poplar_executor",
        "//tensorflow/compiler/plugin/poplar/driver:xla_ipu_common",
    ],
)

popit_cc_library(
    name = "xla_popit_backend",
    srcs = [
        "xla_popit_backend.cc",
    ],
    visibility = ["//tensorflow/compiler/plugin:__subpackages__"],
    deps = [
        "//tensorflow/compiler/plugin/poplar/driver:xla_ipu_common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/stream_executor/lib",
    ],
    alwayslink = True,
)

popit_cc_library(
    name = "xla_popit_device",
    srcs = [
        "xla_popit_device.cc",
    ],
    deps = [
        ":popit_executor",
        ":popit_platform",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/plugin/poplar/driver:xla_ipu_common",
        "//tensorflow/core/kernels:no_op",
    ],
    alwayslink = True,
)

popit_cc_library(
    name = "popit_backend",
    deps = [
        ":popit_compiler",
        ":popit_executable",
        ":popit_executor",
        ":popit_platform",
        ":popit_transfer_manager",
        ":xla_popit_backend",
        ":xla_popit_device",
    ],
)

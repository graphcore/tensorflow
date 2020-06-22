"""Template for the BUILD file for the generated poplar repository
"""

load("@local_config_poplar//poplar:build_defs.bzl", "if_custom_poplibs")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["poplar/bin/popc"],
)

cc_library(
    name = "poplar_headers",
    hdrs = glob(["**/*.hpp"]),
    includes = [
        "gcl/include",
        "poplar/include",
        "poplibs/include",
    ],
)

cc_library(
    name = "poplar_libs",
    srcs = glob(
        [
            "lib*/poplar/libgcl_ct*",
            "lib*/poplar/libpoplar*",
            "lib*/poplar/libtbb.*",
            "lib*/poplar/libtbbmalloc.*",
        ] + if_custom_poplibs([
            "lib*/poplibs/libpoplin*",
            "lib*/poplibs/libpopnn*",
            "lib*/poplibs/libpopops*",
            "lib*/poplibs/libpoprand*",
            "lib*/poplibs/libpopfloat*",
            "lib*/poplibs/libpopsys*",
            "lib*/poplibs/libpoputil*",
        ], [
            "lib*/poplar/libpoplin*",
            "lib*/poplar/libpopnn*",
            "lib*/poplar/libpopops*",
            "lib*/poplar/libpoprand*",
            "lib*/poplar/libpopfloat*",
            "lib*/poplar/libpopsys*",
            "lib*/poplar/libpoputil*",
        ]),
    ),
    deps = [":poplar_headers"],
)

filegroup(
    name = "popsec_headers",
    srcs = glob(["popsec/include/**/*.hpp"]),
)

filegroup(
    name = "popsec_libs",
    srcs = glob(["popsec/lib*/lib*"]),
)

cc_library(
    name = "popsec_lib",
    hdrs = [":popsec_headers"],
    srcs = [":popsec_libs"],
    includes = ["popsec/include"],
)

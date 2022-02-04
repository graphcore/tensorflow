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
    hdrs = glob(["**/*.hpp", "**/*.h"]),
    includes = [
        "gcl/include",
        "poplar/include",
        "poplibs/include",
        "libpvti/include",
        "openmpi/include",
        "popef/include",
        "libpva/include",
        "gccs/include",
    ],
)

cc_library(
    name = "poplar_libs",
    srcs = glob(
        [
            "lib*/poplar/libpoplar*",
        ] + if_custom_poplibs([
            "lib*/poplar/libgcl_ct*",
            "lib*/poplar/libpva.so",
            "lib*/poplar/libpvti.so",
            "lib*/poplar/libtbb.*",
            "lib*/poplar/libtbb_preview.*",
            "lib*/poplar/libtbbmalloc.*",
            "lib*/poplar/libnuma.*",
            "lib*/poplar/libpopef.so",
            "lib*/poplibs/libpoplin*",
            "lib*/poplibs/libpopnn*",
            "lib*/poplibs/libpopops*",
            "lib*/poplibs/libpoprand*",
            "lib*/poplibs/libpopfloat*",
            "lib*/poplibs/libpopsparse*",
            "lib*/poplibs/libpopsys*",
            "lib*/poplibs/libpoputil*",
        ], [
            "lib*/**/libgcl_ct*",
            "lib*/**/libpopef.so",
            "lib*/**/libpva.so",
            "lib*/**/libpvti.so",
            "lib*/**/libtbb.*",
            "lib*/**/libtbb_preview.*",
            "lib*/**/libtbbmalloc.*",
            "lib*/**/libnuma.*",
            "lib*/**/libpoplin*",
            "lib*/**/libpopnn*",
            "lib*/**/libpopops*",
            "lib*/**/libpoprand*",
            "lib*/**/libpopfloat*",
            "lib*/**/libpopsparse*",
            "lib*/**/libpopsys*",
            "lib*/**/libpoputil*",
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

cc_library(
    name = "mpi_lib",
    srcs = glob(["**/libmpi.so"]),
)

filegroup(
    name = "mpirun",
    srcs = glob(["**/mpirun"]),
)

filegroup(
    name = "poprun",
    srcs = glob(["**/poprun"]),
)

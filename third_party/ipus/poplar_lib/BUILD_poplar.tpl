"""Template for the BUILD file for the generated poplar repository
"""

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
    srcs = glob([
        "lib*/**/libgcl_ct*",
        "lib*/**/libpoplar*",
        "lib*/**/libpoplin*",
        "lib*/**/libpopnn*",
        "lib*/**/libpopops*",
        "lib*/**/libpoprand*",
        "lib*/**/libpopfloat*",
        "lib*/**/libpopsys*",
        "lib*/**/libpoputil*",
        "lib*/**/libtbb.*",
        "lib*/**/libtbbmalloc.*",
    ]),
    deps = [":poplar_headers"],
)

cc_library(
    name = "popsec_lib",
    hdrs = glob(["popsec/include/*.hpp"]),
    srcs = glob(["popsec/lib*/lib*"]),
    includes = ["popsec/include"],
)

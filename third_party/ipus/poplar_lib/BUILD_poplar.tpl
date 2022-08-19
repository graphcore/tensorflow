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
        "popit/include",
        "poprithms/include",
        "popdist/include",
        "graphcore_target_access/include",
    ],
)

cc_library(
    name = "poplar_libs",
    srcs = glob(
        [
            "lib*/poplar/libpoplar.so",
            "lib*/poplar/libpoplar_test.so",
            "lib*/**/libpopsolver.so",
            "lib*/popit/libpopit.so",
            "lib*/poprithms/libpoprithms.so",
        ] + if_custom_poplibs([
            "lib*/poplar/libgcl_ct*",
            "lib*/poplar/libpva.so",
            "lib*/poplar/libpvti.so",
            "lib*/poplar/libtbb.*",
            "lib*/poplar/libtbb_preview.*",
            "lib*/poplar/libtbbmalloc.*",
            "lib*/poplar/libnuma.*",
            "lib*/poplar/libpopef.so",
            "lib*/poplibs/libpoplin.so",
            "lib*/poplibs/libpopnn.so",
            "lib*/poplibs/libpopops.so",
            "lib*/poplibs/libpoprand.so",
            "lib*/poplibs/libpopfloat.so",
            "lib*/poplibs/libpopsparse.so",
            "lib*/poplibs/libpopsys.so",
            "lib*/poplibs/libpoputil.so",
            "lib*/popdist/libpopdist.so",
        ], [
            "lib*/**/libgcl_ct*",
            "lib*/**/libpopef.so",
            "lib*/**/libpva.so",
            "lib*/**/libpvti.so",
            "lib*/**/libtbb.*",
            "lib*/**/libtbb_preview.*",
            "lib*/**/libtbbmalloc.*",
            "lib*/**/libnuma.*",
            "lib*/**/libpoplin.so",
            "lib*/**/libpopnn.so",
            "lib*/**/libpopops.so",
            "lib*/**/libpoprand.so",
            "lib*/**/libpopfloat.so",
            "lib*/**/libpopsparse.so",
            "lib*/**/libpopsys.so",
            "lib*/**/libpoputil.so",
            "lib*/**/libpopdist.so",
            "lib*/**/libipu_arch_info.so",
            "lib*/**/libiai_ipu*.so",
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

# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["poplar/bin/popc"],
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.hpp"]),
  includes = ["poplar/include", "poplibs/include"],
)

filegroup(
  name = "poplar_codelets",
  srcs = glob(["**/*gp"]),
)

filegroup(
  name = "poplar_libs",
  srcs = glob([
    "poplar/lib/libpoplar.*",
    "poplibs/lib/libpoplin.*",
    "poplibs/lib/libpopnn.*",
    "poplibs/lib/libpopops.*",
    "poplibs/lib/libpoprand.*",
    "poplibs/lib/libpopfloat.*",
    "poplibs/lib/libpopsys.*",
    "poplibs/lib/libpopsolver.*",
    "poplibs/lib/libpoputil.*",
  ]),
  data = [
      "@local_config_poplar//poplar:poplar_codelets",
  ],
)

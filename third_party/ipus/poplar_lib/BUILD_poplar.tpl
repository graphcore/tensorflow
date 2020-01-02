# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["bin/popc"],
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.hpp"]),
  includes = ["include"],
)

filegroup(
  name = "poplar_codelets",
  srcs = glob(["**/*gp"]),
)

filegroup(
  name = "poplar_libs",
  srcs = glob([
    "lib/libpopfloat.*",
    "lib/libpoplar.*",
    "lib/libpoplin.*",
    "lib/libpopnn.*",
    "lib/libpopops.*",
    "lib/libpoprand.*",
    "lib/libpopsolver.*",
    "lib/libpoputil.*",
  ]),
  data = [
      "@local_config_poplar//poplar:poplar_codelets",
  ],
)


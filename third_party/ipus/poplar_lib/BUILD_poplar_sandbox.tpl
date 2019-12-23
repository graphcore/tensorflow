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

# IMPORTANT: codelets files must be in the same directory as the the Poplar
# libraries at runtime which is why "filegroup" and a custom linker file are
# used instead of glob directly in the "srcs" of cc_library.
cc_library(
  name = "poplar_lib",
  deps = [
  "@local_config_poplar//poplar:poplar_headers",
  ],
  data = [
  "@local_config_poplar//poplar:gen_poplar_linker_options",
  ],
  linkopts = [
      "@$(locations @local_config_poplar//poplar:gen_poplar_linker_options)",
 ],
)

genrule(
  name = "gen_poplar_linker_options",
  srcs= [
      "@local_config_poplar//poplar:poplar_libs",
  ],
  outs = ["poplar_linker_options"],
  cmd = """cat <<EOF > $@
-L`pwd`
$(locations @local_config_poplar//poplar:poplar_libs)
-Wl,-rpath,POPLAR_LIB_DIRECTORY
-Wl,-rpath,POPLIBS_LIB_DIRECTORY
EOF""")


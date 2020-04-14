licenses(["unencumbered"])  # Public Domain or MIT

exports_files(["LICENSE"])

filegroup(
    name = "headers",
    srcs = [
        "include/json/autolink.h",
        "include/json/config.h",
        "include/json/features.h",
        "include/json/forwards.h",
        "include/json/json.h",
        "include/json/reader.h",
        "include/json/value.h",
        "include/json/version.h",
        "include/json/writer.h",
    ],
    visibility = ["//visibility:public"],
    )


cc_library(
    name = "jsoncpp",
    srcs = [
        "include/json/assertions.h",
        "src/lib_json/json_reader.cpp",
        "src/lib_json/json_tool.h",
        "src/lib_json/json_value.cpp",
        "src/lib_json/json_writer.cpp",
    ],
    hdrs = [
    ":headers",
    ],
    copts = [
        "-DJSON_USE_EXCEPTION=0",
        "-DJSON_HAS_INT64",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":private"],
)

cc_library(
    name = "private",
    textual_hdrs = ["src/lib_json/json_valueiterator.inl"],
)

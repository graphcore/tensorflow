licenses(["restricted"])

package(default_visibility = ["//tensorflow/compiler/plugin/poplar:__subpackages__"])

# Rule for generating internal cpp codelets
genrule(
    name = "convert_tf_cpp_graph_sources_to_literal",
    srcs = ["tf.cpp"],
    outs = ["tf.cppembed"],
    cmd = """
        echo 'R"(' > $@
        cat $(SRCS) >> $@
        echo ')"'>> $@
        """,
)

# Rule for generating internal assembly codelets
genrule(
    name = "convert_tf_asm_graph_sources_to_literal",
    srcs = ["tf.S"],
    outs = ["tf.Sembed"],
    cmd = """
        echo 'R"(' > $@
        cat $(SRCS) >> $@
        echo ')"'>> $@
        """,
)

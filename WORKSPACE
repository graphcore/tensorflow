workspace(name = "org_tensorflow")

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

# IPU Specific.
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

new_git_repository(
    name = "horovod_repo",
    remote = "https://github.com/horovod/horovod.git",
    commit = "b52e4b3e6ce5b1b494b77052878a0aad05c2e3ce",
    build_file = "//third_party/ipus/horovod:horovod.BUILD",
)

new_git_repository(
    name = "lbfgspp_repo",
    remote = "https://github.com/yixuan/LBFGSpp.git",
    commit = "f047ef4586869855f00e72312e7b4d78d11694b1",
    build_file = "//third_party/ipus/horovod:lbfgspp.BUILD",
)

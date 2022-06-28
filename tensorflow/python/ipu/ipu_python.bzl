
load("@local_config_popit//:build_defs_popit.bzl", "popit_is_enabled")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

def popit_py_test(**kwargs):
  if popit_is_enabled():
    tf_py_test(**kwargs)
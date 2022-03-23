load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

def poplar_cc_library(**kwargs):
  """ Wrapper for inserting poplar specific build options.
  """
  if not "copts" in kwargs:
    kwargs["copts"] = []

  copts = kwargs["copts"]
  copts.append("-Werror=return-type")

  cc_library(**kwargs)

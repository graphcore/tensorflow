def poplar_cc_library(**kwargs):
  """ Wrapper for inserting poplar specific build options.
  """
  if not "copts" in kwargs:
    kwargs["copts"] = []

  copts = kwargs["copts"]
  copts.append("-Werror=return-type")

  native.cc_library(**kwargs)

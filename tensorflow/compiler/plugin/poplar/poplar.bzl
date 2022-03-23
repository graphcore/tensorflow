
load("@local_config_popit//:build_defs_popit.bzl", "popit_is_enabled")

def poplar_cc_library(**kwargs):
  """ Wrapper for inserting poplar specific build options.
  """
  if not "copts" in kwargs:
    kwargs["copts"] = []

  copts = kwargs["copts"]
  copts.append("-Werror=return-type")

  native.cc_library(**kwargs)


def popit_cc_library(**kwargs):
  """
  Wrapper for building popit libraries. While in developement
  will condtionally insert empty libraries
  """
  popit_enabled = popit_is_enabled()
  if popit_enabled:
    poplar_cc_library(**kwargs)
  else:
    # If not enabled just create an empty cc library
    native.cc_library(
        name = kwargs["name"],
        srcs = [],
    )

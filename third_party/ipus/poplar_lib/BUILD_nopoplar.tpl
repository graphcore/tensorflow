"""Dummy module when building without Poplar support.
"""

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "poplar",
)

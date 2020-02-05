"""Macros for supporting Poplar library
"""

def poplar_available():
    """Returns false because Poplar library was not configured
    """
    return False

def tf_poplar_build_tag():
    """Returns a build tag/hash for displaying along with the Poplar version
    """
    return ""

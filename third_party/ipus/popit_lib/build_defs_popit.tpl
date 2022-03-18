"""Macros for supporting PopIT
"""

def popit_is_enabled():
    """Returns whether popit is enabled"""
    return int({IF_POPIT})

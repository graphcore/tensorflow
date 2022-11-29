# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Configuration utilities
~~~~~~~~~~~~~~~~~~~~~~~
"""

import ast
import collections
import copy
import difflib
from enum import Enum
import inspect
import json
import os
import pydoc
import typing
import sys

from tensorflow.python.eager.context import executing_eagerly
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

if sys.version_info >= (3, 9):
  # Python 3.9 makes breaking changes to ast.Subscript.
  def _get_subscript_slice(subscript_node):
    return subscript_node.slice
else:

  def _get_subscript_slice(subscript_node):
    assert isinstance(subscript_node.slice, ast.Index)
    return subscript_node.slice.value


def _annotation_to_str(node):
  """
  Construct a type hint string from an AST annotation.
  """
  if isinstance(node, str):
    return node
  if isinstance(node, ast.Str):
    return node.s
  if isinstance(node, ast.Attribute):
    return f"{_annotation_to_str(node.value)}.{_annotation_to_str(node.attr)}"
  if isinstance(node, ast.Subscript):
    return f"{_annotation_to_str(node.value)}[{_annotation_to_str(node.slice)}]"
  if isinstance(node, ast.Slice):

    def helper(v):
      return _annotation_to_str(getattr(node, v, ast.Str("")))

    return ":".join(map(helper, ['lower', 'upper', 'step']))
  if isinstance(node, ast.Index):
    return _annotation_to_str(node.value)
  if isinstance(node, ast.Tuple):
    return ', '.join(map(_annotation_to_str, node.elts))
  if isinstance(node, ast.Ellipsis):
    return "..."
  if isinstance(node, ast.Name):
    return node.id
  raise Exception(f"Unhandled {node} when converting type hint to string.")


def _get_type_check_fn_from_AST_type_hints(node, f_globals):
  """
  Function that parses the type hints in an AST AnnAssign node and converts them
  into a callable that checks the type of a value `v` against the type hints.
  Covers basic usage of typing.List, typing.Tuple and typing.Union according to
  the following formal grammar:
  L -> typing.List
  T -> typing.Tuple
  U -> typing.Union
  t -> int | list | tuple | str | float | any python built-in type that can
       be located by pydoc.locate() | any symbol that is available in f_globals

  B -> S | S, B
  S = t | L | L[S] | T | T[B] | T[S, ...] | U[S, B]

  For example, the following are all supported:
  typing.List  # Accept any list
  typing.List[int]  # Accept only a list of integers
  typing.Tuple  # Accept any tuple
  typing.Tuple[int]  # Accept only a tuple with one integer
  typing.Tuple[int, ...]  # Accept only a tuple of integers
  typing.Tuple[int, str, list]  # Accept only a tuple of an int, str and list
  typing.Union[int, typing.List[int]]  # Accept either an int or a list of ints
  int  # Accept only integers
  ClassX  # Accept only 'ClassX' instances.
  some.path.ClassX # Accept only 'ClassX' instances using a qualified path
    within the caller's global context provided by `f_globals'.

  Essentially this function allows you to enforce simple type hints.
  """
  assert isinstance(node, ast.AnnAssign), "Only AnnAssign AST nodes allowed."

  def top_level_type(typ):
    # Strict type comparison over isinstance.
    return lambda v: type(v) == typ, typ  # pylint: disable=unidiomatic-typecheck

  def is_typing_module_attr(node):
    return isinstance(node, ast.Attribute) and \
        isinstance(node.value, ast.Name) and node.value.id == "typing"

  def get_attribute_full_path(node):
    if isinstance(node, ast.Attribute):
      return get_attribute_full_path(node.value) + "." + node.attr
    elif isinstance(node, ast.Name):
      return node.id
    raise ValueError("node is not an ast.Attribute or ast.Name")

  def helper(node):
    # Handle "typing.List", "typing.Tuple"
    if is_typing_module_attr(node):
      if node.attr == "List":
        return top_level_type(list)
      if node.attr == "Tuple":
        return top_level_type(tuple)
      raise Exception(f"Unsupported 'typing' attribute {node.attr}")

    # Handle any arbitrary built-in e.g. 'int', or class/enum names.
    if isinstance(node, ast.Name):
      # Search for the variable in globals() at the time of execution. Failing
      # that, use pydoc to locate it.
      return top_level_type(f_globals.get(node.id, pydoc.locate(node.id)))

    # Subscripts, e.g. X[...]
    if isinstance(node, ast.Subscript):
      lhs = node.value
      if is_typing_module_attr(lhs):
        slice_value = _get_subscript_slice(node)
        # e.g. Union[int, str], check v for all union types
        if lhs.attr == "Union":
          assert isinstance(slice_value, ast.Tuple)
          types = [helper(n) for n in slice_value.elts]
          type_tys = [ty for _, ty in types]
          if int in type_tys and any([issubclass(Enum, ty)
                                      for ty in type_tys]):
            raise ValueError("cannot have a Union of int and Enum: " +
                             "deserialising this attribute may be invalid")
          type_fns = [fn for fn, _ in types]
          return lambda v: any(fn(v) for fn in type_fns), None
        # e.g. Tuple[int, ...], check v elementwise for each type
        if lhs.attr == "Tuple":
          check_tuple = lambda v: isinstance(v, tuple)
          # single element Tuple: check the single element in v for type
          if isinstance(slice_value, ast.Name):
            type_fn, _ = helper(slice_value)
            return lambda v: check_tuple(v) and len(v) == 1 and type_fn(v[
                0]), tuple
          # more than one element Tuple
          if isinstance(slice_value, ast.Tuple):
            # e.g. Tuple[int, ...], check each element in v for the same type
            if len(slice_value.elts) > 1 and isinstance(
                slice_value.elts[1], ast.Ellipsis):
              type_fn, _ = helper(slice_value.elts[0])
              return lambda v: check_tuple(v) and all([type_fn(e)
                                                       for e in v]), tuple
            # e.g. Tuple[int, str], pair-wise (element, type) check
            type_fns = [fn for fn, _ in [helper(n) for n in slice_value.elts]]
            return lambda v: check_tuple(v) and len(v) == len(
                type_fns) and all([fn(e) for fn, e in zip(type_fns, v)]), tuple
        # e.g. List[int], check each element in v for the same type
        if lhs.attr == "List":
          assert not isinstance(
              slice_value,
              ast.Tuple), "List with more than one type not allowed."
          check_list = lambda v: isinstance(v, list)
          type_fn, _ = helper(slice_value)
          return lambda v: check_list(v) and all([type_fn(e) for e in v]), list
        raise Exception(f"Unsupported 'typing' attribute {lhs.attr}")
      raise Exception(
          f"Only 'typing' module types can be subscripted in hints. {lhs}")

    # See if we can resolve this type from the attribute path in the context of
    # the caller, if the type is not a builtin, typing annotation, or addressed
    # simply by name.
    if isinstance(node, ast.Attribute):
      attrib_path = get_attribute_full_path(node)
      return top_level_type(
          f_globals.get(attrib_path, pydoc.locate(attrib_path)))

    raise Exception(f"Unhandled AST node type in type hint {node}")

  return helper(node.annotation)


def _called_from_instance_init(instance, calling_frame):
  """ Helper to check a calling_frame is in an instance's __init__ """
  from_init = inspect.getframeinfo(calling_frame).function == "__init__"
  from_this = calling_frame.f_locals.get('self') is instance
  return from_init and from_this


def _build_full_attribute_name_from_call_stack(init_frame):
  """
  Given a frame in a call stack that points to an assignment which is inside
  a config structure, traverse up the structure by following the assignments to
  build the full name of the attribute relative to the base of the config
  structure. For example, given the following config structure:
  ```
  class NestedConfig(_ConfigBase):
    def __init__(self):
      self.attribute = 1

  class ExampleConfig(_ConfigBase):
    def __init__(self):
      nested_config = NestedConfig(_ConfigBase)
  ```
  And a call frame that points to the `self.attribute = 1` assignment, this
  function goes up the assignment stack, seeing that `NestedConfig` is being
  constructed inside `ExampleConfig` with name `nested_config`, then determining
  the full name of "attribute" to be "nested_config.attribute".
  """
  cur_frame = init_frame
  name_parts = []
  while True:
    # Get the variable name of the current assignment via the AST.
    # We could do it with linecache too but this feels more robust.
    assignment_node = _get_assignment_node_from_call_frame(cur_frame)
    if isinstance(assignment_node, ast.AnnAssign):
      name = assignment_node.target.attr
    else:
      name = assignment_node.targets[0].attr

    name_parts = [name] + name_parts

    # Finish if we reached the root of the nested structure.
    parent_class = cur_frame.f_back.f_locals.get('self', None)
    if not isinstance(parent_class, _ConfigBase):
      break
    # Go up the assignment stack until we reach the config root.
    cur_frame = cur_frame.f_back
  return ".".join(name_parts), len(name_parts)


def _get_docstring_above_AST_node(filename, node):
  """
  Given a frame in a call stack that points to a line in a file's source code,
  find the full docstring above that line, if any.
  For example, if we call this function on a call frame currently pointing to:
  ```
  '''
  A statement that adds 3 to 4
  '''
  3 + 4  <--- call_frame
  ```
  then this function will retrieve the "A statement that adds 3 to 4" docstring.
  If there is no such docstring, "No description provided" is returned.
  """
  # Find the docstring by looking for the first line above the assigning
  # statement with AST nodes associated to it. We have to use the AST node since
  # the docstring could be over a number of source lines and depending on
  # Python version it could be defined at the first or last line of the doc
  # string.
  source_index = _get_source_index(filename)

  nodes = []
  for line_number in reversed(range(0, node.lineno)):
    if line_number in source_index:
      nodes = source_index[line_number]
      break

  # (Docstrings are Exprs with a Str in them in AST)
  if len(nodes) == 2 and isinstance(nodes[0], ast.Expr) and isinstance(
      nodes[0].value, ast.Str):
    return nodes[0].value.s
  return "No description provided."


def _get_assignment_type_and_checker(assign_node, rhs, f_globals):
  """
  Given an AST assignment node, get the type of the RHS of the assignment as
  a string and also build a function that will check a value against the type of
  the RHS. If the assignment node is not annotated, we simply use the type of v.

  Args:
    assign_node: The AST node for the assignment.
    rhs: The right hand side (target) of the assignment as a Python value.
    f_globals: The caller frame's globals, for searching for types within the
      caller's global context.
  """
  # Find possible types...
  if isinstance(assign_node, ast.AnnAssign):
    # ...from Python's type hint annotations
    check_type_fn, attr_type = _get_type_check_fn_from_AST_type_hints(
        assign_node, f_globals)
    # Reconstruct the type hint string from the AST node.
    attr_type_str = _annotation_to_str(assign_node.annotation)
  else:
    # ...from the initial value
    check_type_fn = lambda value: type(value) == type(rhs)  # pylint: disable=unidiomatic-typecheck
    attr_type = type(rhs)
    attr_type_str = attr_type.__name__
  return attr_type, attr_type_str, check_type_fn


_FILENAME_SOURCE_INDEXES = {}


def _get_source_index(filename):
  """
  Helper to get a mapping from the line numbers in a file `filename` to the
  parsed AST nodes for each line. Caches results.
  """
  if filename not in _FILENAME_SOURCE_INDEXES:
    # Get the AST for the file.
    with open(filename, 'r') as f:
      tree = ast.parse(f.read())

    # Create a mapping from AST node line numbers to AST nodes.
    source_index = collections.defaultdict(list)
    for node in ast.walk(tree):
      if hasattr(node, "lineno"):
        source_index[node.lineno].append(node)

      # Also add parent references so we can go up the AST.
      if not hasattr(node, 'parent'):
        node.parent = None
      for child in ast.iter_child_nodes(node):
        child.parent = node

    _FILENAME_SOURCE_INDEXES[filename] = source_index
  return _FILENAME_SOURCE_INDEXES[filename]


def _get_assignment_node_from_call_frame(frame):
  """
  Helper to get the Assign or AnnAssign AST node for a call frame.
  The call frame will point to a specific file and line number, and we use the
  source index to retrieve the AST nodes for that line.
  """
  filename = frame.f_code.co_filename
  # Go up the AST from a node in the call frame line until we find an Assign or
  # AnnAssign, since the (Ann)Assign may be over multiple lines.
  nodes_in_line = _get_source_index(filename).get(frame.f_lineno, [])
  cur_node = nodes_in_line[0]
  while cur_node:
    if isinstance(cur_node, (ast.Assign, ast.AnnAssign)):
      return cur_node
    cur_node = cur_node.parent
  raise Exception("Could not find AST assignment node in the line"
                  f" {filename}:{frame.f_lineno}")


_DEPRECATIONS = {}


def deprecate_config_attribute(name, msg):
  """
  Class decorator to deprecate an attribute in a nested _ConfigBase structure.
  Stores the deprecation in the class's DEPRECATIONS attribute so it can be
  used later when we determine attribute metadata on initialization.

  Args:
    name: The name of the attribute on the _ConfigBase this decorates to
          deprecate.
    msg: The deprecation message to show in documentation and warnings.
  """
  def cls_wrapper(cls):
    _DEPRECATIONS[(cls, name)] = msg
    return cls

  return cls_wrapper


def deprecate_config_attributes(to_deprecate):
  """
  Same as `deprecate_config_attribute` but for multiple at a time.

  Args:
    to_deprecate: a dictionary or list of name: msg pairs
  """
  def cls_wrapper(cls):
    pairs = to_deprecate
    if isinstance(pairs, dict):
      pairs = pairs.items()
    for name, msg in pairs:
      _DEPRECATIONS[(cls, name)] = msg
    return cls

  return cls_wrapper


def running_on_ipu_model():
  """ Check if XLA is configured to run on the ipu model.

  Returns:
    True if XLA is configured to run on the ipu model.
    False if XLA is configured to run on real hardware.
  """
  return "--use_ipu_model" in os.environ.get("TF_POPLAR_FLAGS", "")


class AttributeMetadata:
  def __init__(self,
               name,
               doc="",
               depth=0,
               default=None,
               deprecated_msg=None,
               attr_type=None,
               attr_type_str=None,
               check_type_fn=lambda v: True):
    """
    Encapsulates the metadata for an attribute in a nested _ConfigBase
    structure.

    This docstring is here so it's not shown to users by Sphinx.

    Args:
      name: The full name of the attribute, relative to the structure's root.
      doc: The docstring for the attribute.
      depth: The depth of the attribute in the structure.
      default: The default value for the attribute.
      deprecated_msg: Deprecation message if the attribute is deprecated.
      attr_type: A string describing the allowed types for the attribute.
      check_type_fn: A function that takes in a single value and determines if
                    it's the correct type for this attribute.
    """
    self._name = name
    self.__doc__ = inspect.cleandoc(doc)  # Normalize docstring indentation.
    self._deprecated = deprecated_msg is not None
    self._deprecated_msg = deprecated_msg
    self._type = attr_type_str
    self._actual_type = attr_type
    self._default = default

    self._check_type_fn = check_type_fn
    self._depth = depth

  @property
  def name(self):
    """
    The full name of the option/category, relative to the config structure's
    root.
    """
    return self._name

  @property
  def deprecated(self):
    """
    Whether or not this option/category is deprecated.
    """
    return self._deprecated

  @property
  def deprecated_msg(self):
    """
    The deprecation message for this attribute. `None` if it is not deprecated.
    """
    return self._deprecated_msg

  @property
  def type(self):
    """
    The type of this option, as a string. The type can be a simple Python
    type or a type hint. Categories themselves do not have types.
    """
    return self._type

  @property
  def actual_type(self):
    return self._actual_type

  @property
  def default(self):
    """
    The default value for this option. Categories themselves do not have default
    values.
    """
    return self._default

  def check_type(self, value):
    """
    Checks if `value` is one of the allowed types for this option. Throws a
    TypeError if not.

    Args:
      value: The value to check against this attribute's type.

    Returns:
      True if `value` satisfies this attribute's type.
    """
    if not self._check_type_fn(value):
      raise TypeError(
          f"Trying to set {self.name} to {value}, but it must be of"
          f" type {self.type}")

  def warn_if_deprecated(self):
    """
    Outputs a log warning if this option/category is deprecated.
    """
    if self.deprecated:
      logging.warn(f"{self.name} has been deprecated: {self.deprecated_msg}")

  def _set_deprecated(self, msg):
    self._deprecated = True
    self._deprecated_msg = msg

  def _generate_documentation(self, is_nested_config=False, workaround=False):
    """
    Generate a documentation string for this option/category from its metadata.
    The string is in Sphinx/RST format of the form (comments in brackets):

    ```
    .. _FULL_ATTRIBUTE_NAME: (a referencable label)
    .. py:attribute:: FULL_ATTRIBUTE_NAME
       :type: ATTR_TYPE (a Python type hint annotation)
       :value: ATTR_DEFAULT_VALUE (empty strings display as "")

       .. note:: (optional deprecation note box)

         DEPRECATED: ATTR_DEPRECATED_MSG

       ATTR_DOCSTRING (attribute docstring pasted as-is here)


    ```

    The entire block is indented based on the attribute's "_depth" in the config
    tree.

    Note that due to Sphinx not officially supporting nested attributes, some
    py:attribute:: names must be duplicated. For example, if an attribute's
    full name is "a.b.c.d.e", the "a.b.c" part is duplicated - "a.b.c.a.b.c.d.e"
    This is because Sphinx strips what it thinks is the class name from the
    attribute name. Duplicating it makes the name display correctly after Sphinx
    strips it.
    """
    name = self.name
    if workaround:
      name = name.rsplit('.', 2)[0] + '.' + name
    lines = []

    # Label and attribute domain.
    lines.append(f".. _{self.name}:")
    lines.append(f".. py:attribute:: {name}")

    # Type and default.
    if not is_nested_config:
      default = self.default
      # Note: We want quotes to show in the docs for string types.
      if isinstance(default, str):
        default = '"' + default + '"'
      lines.append(f"   :type: {self.type}")
      lines.append(f"   :value: {default}")
    lines.append("")

    # Deprecation note.
    if self.deprecated:
      lines.append("   .. note::\n")
      lines.append(f"      DEPRECATED: {self.deprecated_msg}\n")

    # Indent and add the docstring.
    for line in self.__doc__.split('\n'):
      lines.append("   " + line)

    # Indent the entire block by the depth.
    return '\n'.join(["   " * (self._depth + 1) + l for l in lines]) + '\n\n'


class _ConfigBase(object):
  """
  A class that can be used to create a user-friendly hierarchical structure of
  attributes that can be converted into a protobuf for the Poplar XLA backend.
  Non-root classes in the structure are hidden from the user so all attributes
  are accessed with chained dot notation from the root class.

  To use, create a root _ConfigBase with some (typed) attributes:
  ```
  class Config(_ConfigBase):
    def __init__(self):
      self.option1: int = 1
      self.option2: typing.Union[int, str] = 2
      self._finalize_base_config()
  ```

  The root can then have _ConfigBase attributes itself, building the hierarchy:
  ```
  class _HiddenCategoryClass(_ConfigBase):
    def __init__(self):
      self.option3 = 3

  class Config(_ConfigBase):
    def __init__(self):
      self.option1: int = 1
      self.option2: typing.Union[int, str] = 2
      self.category1 = _HiddenCategoryClass()
      self._finalize_base_config()

  config = Config()
  print(config.category1.option3)  # 3
  ```

  The structure is user-friendly in the following ways:
    - Deprecated attributes will show warnings when that attribute is set:
    ```
    config.deprecated_option1 = 2
    >>> "deprecation_option1 has been deprecated. Use XXX instead"
    ```

    - If an attribute is mistyped, a similarly spelled one will be suggested:
    ```
    config.optoin1 = 2
    >>> "Did you mean 'option1'?"
    ```

    - Attributes are type-checked on setting, based on either their initial
      value's type or their type hints:
    ```
    config.option2 = True
    >>> "Incorrect type for 'option2', must be..."
    ```

    - New options cannot be created on the structure.
    - Sphinx/RST documentation is automatically generated which reflects the
      structure of the config and contains type, default and deprecation
      information, giving each option a referencable label.
    - Any attribute's metadata can be accessed by users through
      `get_attribute_metadata` and used to e.g. forward TF options to CLI apps.


  Docstrings
  ~~~~~~~~~~

  Config attributes can be given docstrings immediately above their assignment.
  These docstrings are automatically extracted and displayed in the
  documentation, as well as being available to users through the attribute's
  metadata (__doc__). If an attribute isn't given a docstring, it will be
  assigned "No description provided." by default.
  Since nested _ConfigBase attributes are also attributes, they can also be
  given docstrings to e.g. describe those general "categories".


  Types
  ~~~~~
  Basic type hints are supported for non-category attributes. If there are none,
  the type of the initial value will be used. Types are strictly enforced when
  a user sets any attribute. For a full list of supported type hints, see the
  formal grammar in `get_type_check_fn_from_AST_type_hints`.


  Some examples of supported type hints:
  ```
  class _B(_ConfigBase):
    def __init__(self):

      # This can only take integers
      self.a: int = 1

      # This can take either an integer or a string
      self.b: typing.Union[int, str] = '1'

      # This can take either a list of integers or a tuple of integers
      self.c: typing.Union[typing.List[int], typing.Tuple[int, ...]] = (1,)

      # This takes a tuple containing:
      #   - one int/str
      #   - a tuple containing any (non-zero) number of floats and/or bools.
      self.d: typing.Tuple[
                typing.Union[int, str],
                typing.Tuple[
                  typing.Union[float, bool], ...]] = ('1', (1.0, True, False))

      # This can take either a ClassX or a ClassY instance.
      self.e = typing.Union[ClassX, ClassY]
  ```


  Deprecation
  ~~~~~~~~~~~
  Attributes (options and categories) can be deprecated with the
  `deprecate_config_attribute` decorator. To deprecate an attribute "a" in
  a hidden class category "_B", apply the decorator to _B and pass "a" and a
  deprecation message, e.g:
  ```
  @deprecate_config_attribute("a", "'a' will be removed soon.")
  class _B(_ConfigBase):
      ...
  ```
  Deprecation messages will be a warning that looks like:
      x.y.z has been deprecated: DEPRECATION MESSAGE
  The format of this message can be changed in
  `AttributeMetadata.warn_if_deprecated`.
  Deprecation will also appear in the generated documentation.
  Using `deprecate_config_attribute` twice with the same attribute will
  overwrite the first message.
  Deprecation of categories is propagated to all of their children - unless
  those children are already deprecated - with the same message.


  Documentation
  ~~~~~~~~~~~~~

  Sphinx documentation is automatically generated for an entire _ConfigBase
  config from the structure and the docstrings. The nested _ConfigBase *classes*
  are hidden from the user; they are purely an implementation detail used to
  allow the Pythonic interaction with the base config, so all nested configs and
  attributes are listed under the base class in the documentation.
  For each attribute, their types, default value, description and whether or not
  they're deprecated are displayed. For nested configs, the types and default
  value aren't shown.
  Nested attributes at depth d will be indented proportional to d. For this
  reason, avoid making the config structure too deep (both for the width of the
  documentation and for the amount of typing the user has to do).
  Sphinx/RST can be put into an attribute's docstring and the relative
  indentation should be preserved.
  All attributes are given a label that can be referenced. An attribute a.b.c
  can be referenced with :ref:`LINK TEXT <a.b.c>` or :ref:`a.b.c` anywhere in
  the document.

  Finalizing
  ~~~~~~~~~~

  Documentation is generated and deprecation is propagated when
  _ConfigBase._finalize_base_config is called at the end of the base config
  initializer. It doesn't need to be called anywhere else.

  """
  def _get_full_name(self, attr):
    return self._user_attributes.get(attr).name

  def _set_value_of_existing_user_attribute(self, k, v):
    """
    Set the value of an existing user attribute `k` to `v`, with some special
    behaviour:
      - Setting to deprecated user attributes will print a warning.
      - User attributes which are configs can't be set to.
      - The type of `v` is checked against the metadata for `k`.
    This is the setter that the user interacts with.
    """
    assert k in self._user_attributes, f"{k} is not a user attribute."
    # Check the type, deprecation and don't allow setting to configs.
    metadata = self._user_attributes[k]
    metadata.warn_if_deprecated()
    if k in self._nested_configs:
      raise Exception(
          f"{metadata.name} is a category and cannot be assigned to.")
    metadata.check_type(v)
    object.__setattr__(self, k, v)

  def _create_new_user_attribute(self, k, v, caller_frame):
    """
    Add a new user attribute `k` to this config with default value `v`.
    The attribute's metadata is discovered from the `inspect` stack frame it's
    assigned in and the AST of the file it's assigned in.
    This is an internal setter used to build the config.
    """
    # Construct the full name of the attribute in the config structure.
    full_name, depth = _build_full_attribute_name_from_call_stack(caller_frame)

    # Find deprecation through earlier registration on _DEPRECATIONS.
    deprecated_msg = _DEPRECATIONS.get((self.__class__, k), None)

    # Find type string and type checking function from AST node.
    assign_node = _get_assignment_node_from_call_frame(caller_frame)
    attr_type, attr_type_str, type_checker = _get_assignment_type_and_checker(
        assign_node, v, caller_frame.f_globals)

    # Find the docstring above the line of the assignment node.
    filename = caller_frame.f_code.co_filename
    docstring = _get_docstring_above_AST_node(filename, assign_node)

    # Hide default and types for categories.
    default = v
    if isinstance(v, _ConfigBase):
      # Keep track of nested configs for convenience.
      self._nested_configs.append(k)
      default = None
      attr_type = None
      attr_type_str = None

    self._user_attributes[k] = AttributeMetadata(full_name,
                                                 doc=docstring,
                                                 depth=depth,
                                                 default=default,
                                                 deprecated_msg=deprecated_msg,
                                                 attr_type=attr_type,
                                                 attr_type_str=attr_type_str,
                                                 check_type_fn=type_checker)
    object.__setattr__(self, k, v)

  def _maybe_init_internal_state(self):
    if "_user_attributes" not in self.__dict__:
      self.__dict__["_user_attributes"] = {}
      self.__dict__["_nested_configs"] = []

  def __getattr__(self, k):
    """
    __getattr__ is called when __getattribute__ fails to find a match for k in
    the instance's __dict__. Use it to suggest similarly spelled attributes.
    """
    self._maybe_init_internal_state()

    # If we're failing to find a match for the internal state, it has now been
    # initialized, so try again.
    if k in ["_user_attributes", "_nested_configs"]:
      return getattr(self, k)

    suggested = difflib.get_close_matches(k, self._user_attributes.keys())
    sugg_string = f"Did you mean '{self._get_full_name(suggested[0])}'?" \
        if suggested else ""
    raise ValueError(
        f"'{k}' is not a valid attribute of this config. {sugg_string}")

  def __setattr__(self, k, v):
    """
    Override setter to redirect to user-facing attribute setters + creators.
    """
    self._maybe_init_internal_state()

    # Setting the value of an existing user-facing attribute.
    if k in self._user_attributes:
      self._set_value_of_existing_user_attribute(k, v)
      return

    # Only allow creating new attributes in the __init__ of this class.
    caller_frame = inspect.currentframe().f_back
    if k not in self.__dict__:
      if not _called_from_instance_init(self, caller_frame):
        # Call the failing getter for a suggestion and exception.
        self.__getattr__(k)

    # Adding a new user-facing attribute.
    self._create_new_user_attribute(k, v, caller_frame)

  def _to_protobuf(self, pb):
    """
    Convert nested _ConfigBases to a protobuf.
    If an inheritor wants to modify the protobuf, it must implement this method.
    If it also contains nested configs, it should call this method too to
    recurse into the nested configs to allow them to modify the protobuf too.
    """
    for config_name in self._nested_configs:
      getattr(self, config_name)._to_protobuf(pb)  # pylint: disable=protected-access

  def to_dict(self):
    """
    Export the configuration stored within this configuration object to a dict.

    Returns:
      A dictionary containing the configuration.
    """
    dct = {}
    # Iterate over the user-facing attributes and their metadata.
    for name, metadata in self._user_attributes.items():
      attr = getattr(self, name)
      # Recurse into nested configs and save normal attributes.
      if name in self._nested_configs:
        dct.update(attr.to_dict())
      else:
        # Convert enums to ints.
        attr = attr.value if isinstance(attr, Enum) else attr
        dct[metadata.name] = attr
    return dct

  def from_dict(self, dct):
    """
    Restore configuration from a dict object.

    Args:
      dct: A dictionary containing a configuration.
    """
    # Iterate over the user-facing attributes and their metadata.
    for name, metadata in self._user_attributes.items():
      attr = getattr(self, name)
      # Recurse into nested configs and restore normal attributes.
      if name in self._nested_configs:
        attr.from_dict(dct)
      elif metadata.name in dct:
        val = dct[metadata.name]
        # Convert ints to enums, using a simple check to see if the default
        # value was also an Enum.
        typ = metadata.actual_type
        val = attr.__class__(val) if typ is not None and issubclass(
            typ, Enum) else val
        if val != attr:
          setattr(self, name, val)
      else:
        logging.warn(
            f"{metadata.name} didn't have a value: the existing value "
            "will be retained")

  def to_json(self):
    """
    Export the configuration stored within this configuration object as a
    JSON string.

    Returns:
      A JSON string containing the configuration.
    """
    return json.dumps(self.to_dict())

  def from_json(self, json_cfg):
    """
    Restore configuration from a JSON string.

    Args:
      json_cfg: A JSON string containing a configuration.
    """
    return self.from_dict(json.loads(json_cfg))

  def _finalize_base_config(self, _root_class=None, _parent_metadata=None):
    """
    Propagate deprecation and generate the root class docstring.
    - If a sub-category is deprecated, then everything underneath it is also
      automatically deprecated, if it isn't already, with the same message.
    - The root class's docstring is modified with automatically generated
      documentation based on the inline docstrings for every attribute in its
      structure.
    _root_class and _parent_metadata are internal helper arguments used in the
    recursion and shouldn't be used directly.
    This function should be called at the end of the __init__ for the root
    class of the config structure only.
    """
    # Make this function idempotent on __doc__. Otherwise, instantiating the
    # root class twice would duplicate its docstring.
    if not _root_class:
      self.__class__.__doc__ = ""

    for i, (config_attr, metadata) in enumerate(self._user_attributes.items()):
      is_nested_config = config_attr in self._nested_configs

      # Don't document private attributes (this includes internal ones).
      if config_attr.startswith('_'):
        continue

      # Propagate parent config deprecation to its child if the child isn't
      # already deprecated.
      if _parent_metadata and _parent_metadata.deprecated and \
          not metadata.deprecated:
        metadata._set_deprecated(_parent_metadata.deprecated_msg)  # pylint: disable=protected-access

      # Generate docs
      root_class = _root_class or self.__class__

      # Note: Sphinx does not officially support nested attributes,
      # (https://github.com/sphinx-doc/sphinx/issues/9099), so the first nested
      # attribute at each depth won't display its full name correctly for
      # attributes nested deeper than two. We work around this by duplicating
      # the part of the name that Sphinx strips.
      # pylint: disable=protected-access
      workaround = i == 0 and metadata._depth > 2
      root_class.__doc__ += metadata._generate_documentation(
          is_nested_config, workaround=workaround)
      # pylint: enable=protected-access

      # Go deeper into nested configs.
      if is_nested_config:
        getattr(self, config_attr)._finalize_base_config(root_class, metadata)  # pylint: disable=protected-access

  def get_attribute_metadata(self, attr):
    """
    Get the attribute metadata for `attr`.

    Args:
      attr: required, a string which specifies which attribute to retrieve
            metadata for. Must be its full name relative to the category
            this method is being called on.
    Returns:
      An :py:class:`~tensorflow.python.ipu.config.AttributeMetadata` object
      containing the metadata for the attribute.
    """
    try:
      parts = attr.split('.')
      config = self
      for part in parts[:-1]:
        config = getattr(config, part)
      # Check that it exists, suggesting similar attributes if not.
      getattr(config, parts[-1])
      return config._user_attributes[parts[-1]]  # pylint: disable=protected-access
    except (IndexError, ValueError, KeyError) as e:
      raise ValueError(f"Could not get attribute metadata for '{attr}': {e}")

  def __deepcopy__(self, memo):
    cls = self.__class__
    new = cls.__new__(cls)
    memo[id(self)] = new
    for k, v in self.__dict__.items():
      new.__dict__[k] = copy.deepcopy(v, memo)
    return new

  def __str__(self):
    def build_line(name, metadata):
      attr = getattr(self, name)
      # Recurse into nested configs.
      if name in self._nested_configs:
        return str(attr)

      val = attr if isinstance(attr, Enum) else repr(attr)
      return f"{metadata.name} = {val}"

    return os.linesep.join(
        build_line(n, m) for n, m in self._user_attributes.items())

  def __repr__(self):
    indented = ["  " + line for line in str(self).split(os.linesep)]
    return os.linesep.join([f"{self.__class__.__name__}:", *indented])


def _poplar_options_to_protobuf(opts, pb_target):
  """
  Populate a protobuf field `pb_target` with the options from an options
  dictionary `opts`.
  """
  for option_name, value in opts.items():
    opt = pb_target.add()
    opt.option = option_name
    opt.value = value


class SelectionOrder(Enum):
  """Depending on the communication pattern of the model, the order in
  which the IPUs are selected and mapped to shards can impact the performance.

  For example, given a model which executes on multiple IPUs:

  .. code-block:: python

    def sharded_graph(pa, pb, pc, pd):
      with ipu.scopes.ipu_shard(0):
        o1 = pa + pb
      with ipu.scopes.ipu_shard(1):
        o2 = o1 + pc
      with ipu.scopes.ipu_shard(2):
        o3 = o2 + pd
        return o3

  and a Graphcore Pod system with 16 IPUs:

  .. code-block:: none

     _______               _______
    |       |             |       |
    |  14   |=============|  15   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |  12   |=============|  13   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |  10   |=============|  11   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   8   |=============|   9   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   6   |=============|   7   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   4   |=============|   5   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   2   |=============|   3   |
    |_______|             |_______|
        ||                    ||
     _______               _______
    |       |             |       |
    |   0   |=============|   1   |
    |_______|             |_______|

  Here, each numbered square represents an IPU with the given device ID and the
  `==` and `||` connections represent IPUs directly connected via IPU-Links.

  We can see that the `ipu_shard(0)` directly communicates with `ipu_shard(1)`
  and that `ipu_shard(1)` directly communicates with `ipu_shard(2)`.

  If the shards 0, 1, 2 were mapped to IPUs 0, 1, 2 in that order, then the
  communication between shards 1 and 2 would not have a direct connection via an
  IPU-Link and would have to perform a "hop" through an intermediate IPU.

  If the shards 0, 1, 2 were mapped to IPUs 0, 1, 3 in that order, then the
  communication between shards 1 and 2 would have a direct connection via an
  IPU-Link, which will reduce the communication cost.

  This enumeration is used to control the order in which the IPUs are selected.
  Currently, the following IPU selection orderings are supported:

  * `AUTO`: automatically try and select the best selection given the network.
  * `ZIGZAG`: follow the natural ordering of IPUs. In the above example, the
    IPUs would be selected in the following order:
    `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`.
  * `SNAKE`: select IPUs such that each consecutive shard is directly
    connected via IPU-Links to the shard before and after. In the above
    example, the IPUs would be selected in the following order:
    `0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14`.
  * `HOOF`: select IPUs such that each consecutive shard is directly
    connected via IPU-Links to the shard before and after, and the last and
    first shard are on adjacent IPUs. In the above example, the IPUs would be
    selected in the following order:
    `0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3, 1`.

  The `SNAKE` and `HOOF` IPU selection orders are particularly beneficial for
  pipelined models.
  """
  AUTO = config_pb2.IpuSelectionOrder.Value("AUTO")
  ZIGZAG = config_pb2.IpuSelectionOrder.Value("ZIGZAG")
  SNAKE = config_pb2.IpuSelectionOrder.Value("SNAKE")
  HOOF = config_pb2.IpuSelectionOrder.Value("HOOF")


class ExecutionProfileType(Enum):
  """The execution profile type indicates the desired information in the
  execution profile.

  * `NO_PROFILE` indicates that there should be no execution profiling.
  * `DEVICE_PROFILE` indicates that the execution profile should contain only
    device wide events.
  * `IPU_PROFILE` indicates that the profile should contain IPU level
    execution events.
  * `TILE_PROFILE` indicates that the profile should contain Tile level
    execution events.
  """
  NO_PROFILE = config_pb2.IpuExecutionProfileType.Value("NO_PROFILE")
  DEVICE_PROFILE = config_pb2.IpuExecutionProfileType.Value("DEVICE_PROFILE")
  IPU_PROFILE = config_pb2.IpuExecutionProfileType.Value("IPU_PROFILE")
  TILE_PROFILE = config_pb2.IpuExecutionProfileType.Value("TILE_PROFILE")


class DeviceConnectionType(Enum):
  """Enumeration to describe the mechanism used to attach to the Poplar
  device.

  * `ALWAYS` indicates that the system will attach when configuring the
    device.
  * `ON_DEMAND` will defer connection to when the IPU is needed.
  * `PRE_COMPILE` will never try to attach to a device and anything which is
    meant to be executed on the device will return all zeros. Used to
    pre-compile Poplar programs on machines without IPUs. For more information,
    see :ref:`precompiling_executables`.
  * `NEVER` will never try to attach to a device.
  """
  ALWAYS = config_pb2.IpuDeviceConnectionType.ALWAYS
  ON_DEMAND = config_pb2.IpuDeviceConnectionType.ON_DEMAND
  PRE_COMPILE = config_pb2.IpuDeviceConnectionType.PRE_COMPILE
  NEVER = config_pb2.IpuDeviceConnectionType.NEVER


class MergeRemoteBuffersBehaviour(Enum):
  """The remote buffers merging behaviour indicates when or if compatible remote
  buffers should be merged.

  * `NO_MERGING` indicates that there should be no merging.
  * `MERGE` indicates that all compatible remote buffers will be merged.
  * `IF_BENEFICIAL` indicates that compatible remote buffers will only
    be merged when it is considered beneficial for code re-use.
  """
  NO_MERGING = threestate_pb2.ThreeState.Value("THREESTATE_OFF")
  MERGE = threestate_pb2.ThreeState.Value("THREESTATE_ON")
  IF_BENEFICIAL = threestate_pb2.ThreeState.Value("THREESTATE_UNDEFINED")


class SchedulingAlgorithm(Enum):
  """Controls the algorithm that the scheduler uses.

  * `CHOOSE_BEST` compares several of the scheduling algorithms below and
    selects the one that leads to the lowest predicted overall peak liveness.
    This can sometimes produce incorrect results because the overall peak
    liveness isn't always a good measure for the maximum liveness on one tile of
    the processor.
  * `CLUSTERING` groups clusters of operations together in order to look through
    stretches of instructions with potentially high liveness.
  * `POST_ORDER` schedules the instructions in the order which is obtained by
    walking the graph in 'post order'.
  * `LOOK_AHEAD` looks ahead a number of operations from any schedulable one, as
    given by the :ref:`maximum scheduler lookahead depth
    <scheduling.maximum_scheduler_lookahead_depth>` and :ref:`maximum scheduler
    search space size <scheduling.maximum_scheduler_search_space_size>` options.
    It attempts to look through areas of high liveness.
  * `SHORTEST_PATH` gives priority to the shortest path to the root.
  """
  CHOOSE_BEST = config_pb2.IpuSchedulingAlgorithm.Value("CHOOSE_BEST")
  CLUSTERING = config_pb2.IpuSchedulingAlgorithm.Value("CLUSTERING")
  POST_ORDER = config_pb2.IpuSchedulingAlgorithm.Value("POST_ORDER")
  LOOK_AHEAD = config_pb2.IpuSchedulingAlgorithm.Value("LOOK_AHEAD")
  SHORTEST_PATH = config_pb2.IpuSchedulingAlgorithm.Value("SHORTEST_PATH")


# pylint: disable=pointless-string-statement
class _MultiReplicaDistributionConfig(_ConfigBase):
  def __init__(self):
    """
    The index of the current process being configured.
    """
    self.process_index = 0
    """
    The total number of processes. When set to 0 (default), multi-replica
    distribution will not be used.
    """
    self.process_count = 0

  def _to_protobuf(self, pb):
    if running_on_ipu_model() and self.process_count > 0:
      raise Exception(
          "Multi-replica distribution is not supported on the IPU model.")

    if self.process_count and not (0 <= self.process_index <
                                   self.process_count):
      raise ValueError(
          f"{self._get_full_name('process_index')} must be in the range"
          f" [0, {self._get_full_name('process_count')}).")

    pb.multi_replica_process_index = self.process_index
    pb.multi_replica_process_count = self.process_count


class _ExperimentalConfig(_ConfigBase):
  def __init__(self):
    """
    The data which is streamed to/from the device might be stored in different
    layouts on the device and on the host. If so, rearrangement is performed on
    the device by default. By enabling this option the rearrangement will be
    performed on the host at the expense of latency.
    """
    self.always_rearrange_copies_on_the_host = False
    """
    When set to true,
    :py:class:`~tensorflow.python.ipu.embedding_ops.HostEmbedding` will make use
    of Poplar remote buffers. The creation of this remote buffer may take
    several minutes. The remote buffer will be synchronised with every IPU
    execution, so we recommend that you use high :ref:`steps_per_execution
    <using-steps-per-execution>` with this option.
    """
    self.enable_remote_buffer_embedding = False
    """
    Enable prng seed management. This aims to reduce divergence of weights
    when running models across multiple replicas with stochastic rounding.
    """
    self.enable_prng_stability = False
    """
    Sub-category containing configuration options controlling multi replica
    distribution. This will use the Poplar runtime replica subset feature to let
    multiple processes collaborate on executing the same Poplar program by
    executing a subset of the global replicas each.

    The total global replication factor will be equal to the local replication
    factor multiplied by the :ref:`process_count
    <experimental.multi_replica_distribution.process_count>`.
    """
    self.multi_replica_distribution = _MultiReplicaDistributionConfig()

  def _to_protobuf(self, pb):
    pb.speed_size_config.always_rearrange_copies_on_the_host = \
        self.always_rearrange_copies_on_the_host
    pb.enable_experimental_remote_buffer_embedding = \
        self.enable_remote_buffer_embedding
    pb.enable_experimental_prng_stability = \
        self.enable_prng_stability

    # Go deeper into nested configs.
    super()._to_protobuf(pb)  # pylint: disable=protected-access


class StochasticRoundingBehaviour(Enum):
  """ Controls how stochastic rounding is performed.

  `OFF` disables stochastic rounding.
  `ON` enables stochastic rounding.
  `REPLICA_IDENTICAL_ONLY` enables stochastic rounding for portions of the graph
  which are identified as being replica identical - meaning that when executed
  with replication they produce the same result on each replica.
  """
  @staticmethod
  def from_bool(value):
    return StochasticRoundingBehaviour.ON if value \
      else StochasticRoundingBehaviour.OFF

  OFF = config_pb2.StochasticRoundingBehaviour.Value("StochasticRounding_Off")
  ON = config_pb2.StochasticRoundingBehaviour.Value("StochasticRounding_On")
  REPLICA_IDENTICAL_ONLY = config_pb2.StochasticRoundingBehaviour.Value(
      "StochasticRounding_ReplicaIdenticalOnly")


class _FloatingPointBehaviourConfig(_ConfigBase):
  def __init__(self):
    """
    If True, a floating point invalid operation (defined by IEEE 754)
    will cause an exception.
    """
    self.inv = False
    """
    If True, a floating point divide by zero operation will cause an exception.
    """
    self.div0 = False
    """
    If True, a floating point overflow will cause an exception.
    """
    self.oflo = False
    """
    A :py:class:`~tensorflow.python.ipu.config.StochasticRoundingBehaviour`.
    If `StochasticRoundingBehaviour.OFF` (default) then stochastic rounding
    will be disabled. Otherwise it's enabled with the semantics of the
    particular option.
    """
    self.esr = StochasticRoundingBehaviour.OFF
    """
    If True, Not-a-Number (NaN) on overflow mode will be enabled.
    """
    self.nanoo = False
    """
    If True, unconditionally enables all floating point behaviour options
    (:ref:`inv <floating_point_behaviour.inv>`,
    :ref:`div0 <floating_point_behaviour.div0>`,
    :ref:`oflo <floating_point_behaviour.oflo>`,
    :ref:`esr <floating_point_behaviour.esr>`,
    :ref:`nanoo <floating_point_behaviour.nanoo>`) when the IPUConfig is
    configured.
    """
    self.set_all = False

  def _to_protobuf(self, pb):
    for opt in ['inv', 'div0', 'oflo', 'nanoo']:
      val = getattr(self, opt) if not self.set_all else True
      setattr(pb.floating_point_behaviour, opt, val)

    if self.set_all:
      pb.floating_point_behaviour.esr = StochasticRoundingBehaviour.ON.value
    else:
      esr = self.esr
      pb.floating_point_behaviour.esr = esr.value


class _IOTilesConfig(_ConfigBase):
  def __init__(self):
    """
    Number of tiles to reserve for I/O.
    """
    self.num_io_tiles = 0
    """
    Whether to place TensorFlow I/O operations on the I/O tiles.
    """
    self.place_ops_on_io_tiles = False
    """
    Proportion of I/O tiles' memory which can be used to store data in, with the
    remaining memory assumed to be used by code. If the size of data which is to
    be stored on I/O tiles exceeds the total I/O tiles memory multiplied by this
    proportion, then a warning message will appear and the operations will not
    be placed on I/O tiles.
    """
    self.available_memory_proportion = 0.9

  def _to_protobuf(self, pb):
    if self.place_ops_on_io_tiles and self.num_io_tiles == 0:
      raise ValueError("Cannot place ops on I/O tiles when"
                       f" {self._get_full_name('num_io_tiles')} == 0")

    pb.num_io_tiles = self.num_io_tiles
    pb.place_ops_on_io_tiles = self.place_ops_on_io_tiles
    pb.io_tile_available_memory_proportion = self.available_memory_proportion


class _IPUDeviceConnectionConfig(_ConfigBase):
  def __init__(self):
    """
    Configure when to attach to the device. For example, you can use this to
    compile and cache a program without attaching to an IPU, and then later run
    on a real IPU device without recompiling. Setting the connection type
    doesn't impact the ability to profile a model. For possible values, see
    :py:class:`~tensorflow.python.ipu.config.DeviceConnectionType`.

    .. code-block:: python

      # Compile without attaching to the device.
      config = IPUConfig()
      config.device_connection.type = DeviceConnectionType.ON_DEMAND
    """
    self.type = DeviceConnectionType.ALWAYS
    """
    Version of the IPU hardware to use (string). Must be one of "ipu1", "ipu2"
    or "" (default). Only required if the
    :ref:`connection type <device_connection.type>` provided is
    `DeviceConnectionType.PRE_COMPILE` or `DeviceConnectionType.NEVER`.
    """
    self.version = ""
    """
    Default to `False`. When :ref:`connection type <device_connection.type>` is
    `DeviceConnectionType.PRE_COMPILE`, `DeviceConnectionType.NEVER` or
    `DeviceConnectionType.ON_DEMAND`, this argument is used to indicate whether
    remote buffers are enabled and supported in the system which will eventually
    be used to execute the compiled programs. Set it to True if the system on
    which you will execute the compiled programs has remote buffers enabled and
    `connection_type` is not `DeviceConnectionType.ALWAYS`. If the
    `connection_type` is `DeviceConnectionType.ALWAYS` then the
    `enable_remote_buffers` parameter is ignored because in that case it is
    possible to query the device and check if remote buffers are supported on
    it (if they are, they will be used automatically).

    In order to check whether your target system supports remote buffers you can
    run the command:

    .. code-block:: console

      $ gc-info -d 0 -i | grep "remote buffers supported:"

    If you see ``remote buffers supported: 1`` in the output, that means that
    remote buffers are supported on your system. For more information, see the
    `gc-info documentation
    <https://docs.graphcore.ai/projects/command-line-tools/en/latest/gc-info_main.html>`_.
    """
    self.enable_remote_buffers = False

  def _to_protobuf(self, pb):
    if self.type in [
        DeviceConnectionType.PRE_COMPILE, DeviceConnectionType.NEVER
    ] and self.version == "":
      raise ValueError(
          f"{self._get_full_name('version')} must be set when"
          f" {self._get_full_name('type')} is DeviceConnectionType.NEVER or"
          " DeviceConnectionType.PRE_COMPILE")

    pb.device_connection_type = self.type.value
    pb.ipu_version = self.version
    pb.enable_remote_buffers_without_device = self.enable_remote_buffers


class _IPUModelConfig(_ConfigBase):
  def __init__(self):
    """
    Whether or not to compile IPU code for modelling.
    """
    self.compile_ipu_code = True
    """
    The number of tiles per IPU Model device. When set to 0 (the default),
    Poplar will use the standard number of tiles for the chosen
    :ref:`version <ipu_model.version>`.
    """
    self.tiles_per_ipu = 0
    """
    Specify the IPU version to be used by the IPU Model. Options are "ipu1" or
    "ipu2" (default).
    """
    self.version = "ipu2"

  def _to_protobuf(self, pb):
    if self.version == "" and running_on_ipu_model():
      raise ValueError(
          f"{self._get_full_name('version')} must be set when using the"
          " IPUModel.")

    pb.ipu_model_config.compile_ipu_code = self.compile_ipu_code
    pb.ipu_model_config.tiles_per_ipu = self.tiles_per_ipu
    pb.ipu_model_config.ipu_model_version = self.version


class _IpuAlgebraicSimplifierConfig(_ConfigBase):
  def __init__(self):
    """
    Enables optimizations which allow arbitrary reassociations and
    transformations of mathematical operations with no accuracy guarantees.
    Enabling this option can result in incorrect output for programs that depend
    on an exact implementation of IEEE floating point for maths functions. It
    may, however, yield faster code for programs that do not require the
    guarantees of these specifications.
    """
    self.fast = False
    """
    Enable dot strength optimization. When set to True, the graph optimizer
    will convert a dot product where either the LHS or the RHS contains only
    batch and/or contracting dimensions to an elementwise matrix
    multiplication.
    """
    self.dot_strength = True

  def _to_protobuf(self, pb):
    pb.algebraic_simplifier_config.enable_fast_math\
      = self.fast
    pb.algebraic_simplifier_config.enable_dot_strength\
      = self.dot_strength


class _MatmulConfig(_ConfigBase):
  def __init__(self):
    """
    Controls whether or not the "Pass" type of the MatMul is passed to PopLibs.
    When set to True, PopLibs will not be told about the type of the MatMuls in
    the graph. This can save memory in some circumstances, such as large batch
    ResNet models. See `matMul` in the PopLibs API reference.
    """
    self.clear_pass_type = False
    """
    Set the PopLibs matrix multiplication options for the session. Must be a
    dictionary of valid PopLibs matrix multiplication options. See `matMul` in
    the PopLibs API reference for the full list of options. The options will be
    applied to all matmul operations in the session graph during compilation.

    Of particular note is the `availableMemoryProportion` parameter which is
    the amount of memory allocated for use for temporary data whilst the
    operation is executing (for example, for intermediate calculated values or
    temporary values passed between tiles on the IPU). The value is specified
    as a proportion of available memory on the IPU. So, for example, a value of
    0.1 will constrain the library to use 10% of the total memory for temporary
    data.

    See the technical note on `Optimising Temporary Memory Usage for
    Convolutions and Matmuls on the IPU
    <https://docs.graphcore.ai/projects/available-memory/>`_ for more details and for some practical
    examples of using `availableMemoryProportion`.

    Another important parameter is `partialsType`, which sets the type of the
    values of intermediate calculations (partials). This parameter can either be
    set to `"float"` (for `float32`) or `"half"` (for `float16`). Note the use
    of `"float"` or `"half"` and not `"float32"` or `"float16"` for the
    parameter values (this is because Poplar/PopLibs uses the IEEE definitions
    of what the datatypes should be called). An example showing how to use this
    parameter is shown below:

    .. code-block:: python

      cfg = config.IPUConfig()
      cfg.matmuls.poplar_options['partialsType'] = "half"
      cfg.configure_ipu_system()
    """
    self.poplar_options = {}

  def _to_protobuf(self, pb):
    pb.clear_matmul_pass_type = self.clear_pass_type
    _poplar_options_to_protobuf(self.poplar_options, pb.matmul_options)


class _ConvolutionConfig(_ConfigBase):
  def __init__(self):
    """
    Set the PopLibs convolution options for the session. Must be a dictionary of
    valid PopLibs convolution options. See `createWeights` in the PopLibs API
    reference for the full list of options. The options will be applied to all
    convolution operations in the session graph during compilation.

    Of particular note is the `availableMemoryProportion` parameter which is
    the amount of memory allocated for use for temporary data whilst the
    operation is executing (for example, for intermediate calculated values or
    temporary values passed between tiles on the IPU). The value is specified
    as a proportion of available memory on the IPU. So, for example, a value of
    0.1 will constrain the library to use 10% of the total memory for temporary
    data.

    See the technical note on `Optimising Temporary Memory Usage for
    Convolutions and Matmuls on the IPU
    <https://docs.graphcore.ai/projects/available-memory/>`_ for more details and for some practical
    examples of using `availableMemoryProportion`.

    Another important parameter is `partialsType`, which sets the type of the
    values of intermediate calculations (partials). This parameter can either be
    set to `"float"` (for `float32`) or `"half"` (for `float16`). Note the use
    of `"float"` or `"half"` and not `"float32"` or `"float16"` for the
    parameter values (this is because Poplar/PopLibs uses the IEEE definitions
    of what the datatypes should be called). An example showing how to use this
    parameter is shown below:

    .. code-block:: python

      cfg = config.IPUConfig()
      cfg.convolutions.poplar_options['partialsType'] = "half"
      cfg.configure_ipu_system()
    """
    self.poplar_options = {}

  def _to_protobuf(self, pb):
    _poplar_options_to_protobuf(self.poplar_options, pb.convolution_options)


class _SliceConfig(_ConfigBase):
  def __init__(self):
    """
    Set the PopLibs slice options for the session. Must be a dictionary of valid
    PopLibs slice options. See `embedding::plan` in the PopLibs API reference
    for the full list of options. The options will be passed to multiSlice,
    multiUpdate, and multiUpdateAdd poplibs calls. These are most commonly
    generated when using embeddings.

    Of particular note is the `availableMemoryProportion` parameter which is
    the amount of memory allocated for use for temporary data whilst the
    operation is executing (for example, for intermediate calculated values or
    temporary values passed between tiles on the IPU). The value is specified
    as a proportion of available memory on the IPU. So, for example, a value of
    0.1 will constrain the library to use 10% of the total memory for temporary
    data.
    """
    self.poplar_options = {}

  def _to_protobuf(self, pb):
    _poplar_options_to_protobuf(self.poplar_options, pb.slice_options)


class _PoolingConfig(_ConfigBase):
  def __init__(self):
    """
    Set the PopLibs pooling compilation options for the session. Must be a
    dictionary of valid PopLibs pooling options. See `pool` in the PopLibs API
    reference for the full list of options. The options will be applied to all
    pooling operations in the session graph during compilation.
    """
    self.poplar_options = {}

  def _to_protobuf(self, pb):
    _poplar_options_to_protobuf(self.poplar_options, pb.pooling_options)


class _NormsExperimentalConfig(_ConfigBase):
  def __init__(self):
    """
    When executing fused batch-norms for training, this option specifies
    how many replicas to aggregate the batch statistics across. For example, if
    a model is being executed across four replicas and this option is set to
    two, replicas 0 and 1 will be grouped together and replicas 2 and 3 will be
    grouped together and the batch norm statistics will be synchronously
    all-reduced every time the layer is executed (including any recomputation)
    across the replicas within a group. This option should not be used when
    using model parallelism (pipelining) and it is not supported with I/O tiles.
    When recomputation is enabled and the training fused batch norm operation is
    recomputed, the statistics will have to be all-reduced again, unless the
    :py:attr:`~tensorflow.python.ipu.ops.pipelining_ops
    .RecomputationMode.RecomputeAndBackpropagateInterleaved`
    recomputation mode is used.
    """
    self.distributed_batch_norm_replica_group_size = 1

  def _to_protobuf(self, pb):
    if self.distributed_batch_norm_replica_group_size < 1:
      raise ValueError(
          f"{self._get_full_name('distributed_batch_norm_replica_group_size')}"
          " must be at least 1.")

    pb.experimental_distributed_batch_norm_replica_group_size = \
        self.distributed_batch_norm_replica_group_size


class _NormConfig(_ConfigBase):
  def __init__(self):
    """
    If True, computes the mean minus the activations first before
    computing the variance. The implementation with this flag set to True is
    slower than when set to False.
    """
    self.use_stable_statistics = False
    """
    Sub-category containing experimental configuration options for
    normalizations that may be changed or removed with short or no notice.
    """
    self.experimental = _NormsExperimentalConfig()

  def _to_protobuf(self, pb):
    pb.use_stable_norm_statistics = self.use_stable_statistics

    # Go deeper into nested configs.
    super()._to_protobuf(pb)  # pylint: disable=protected-access


@deprecate_config_attribute(
    "enable_fast_math",
    "'enable_fast_math' has been moved to 'optimizations.math.fast'. "
    "It will be removed from this location in a future release.")
class _OptimizationConfig(_ConfigBase):
  def __init__(self):
    """
    Sub-category containing configuration options related to simplifying
    algebraic mathematical expressions..
    """
    self.math = _IpuAlgebraicSimplifierConfig()
    """
    If True (default), prefetching of data for data streams on the host will be
    overlapped with execution on the IPU.
    """
    self.prefetch_data_streams = True
    """
    If True, fuse embedding lookups which are on the same tensor. This might
    improve performance but increase memory usage.
    """
    self.combine_embedding_lookups = False
    """
    If True, fuse matmul operations if they share the same weights or the same
    input.
    """
    self.combine_matmuls = False
    """
    If True (default), operations in the graph which are the same but with
    different input tensors may be outlined. This means the same code will be
    re-used to execute them, reducing the amount of program code, but their
    inputs will be exchanged into a common memory location to do so, increasing
    execution time. If you care more about speed than memory, these
    optimizations can be disabled by setting this option to False.
    """
    self.enable_graph_outlining = True
    """
    If True, this flag will merge the streamed host to device input copies into
    one larger copy.  This may reduce the time to copy data from the host, at
    the expense of increasing the live tensor memory on the device.
    """
    self.merge_infeed_io_copies = True
    """
    The maximum number of bytes that can be waiting before a cross replica sum
    op is scheduled. 0 (default) means that they are scheduled immediately.
    This value represents an always-live vs not-always-live trade off -
    increasing the max_cross_replica_sum_buffer_size will lead to larger
    temporary buffers in the cross replica sums, but fewer cross replica sums
    overall and therefore less control code. If your model contains a lot of
    trainable variables, then it is strongly advised to consider adjusting this
    option.
    """
    self.maximum_cross_replica_sum_buffer_size = 0
    """
    The maximum number of bytes that can be waiting before a reduce scatter op
    is scheduled.
    """
    self.maximum_reduce_scatter_buffer_size = 0
    """
    The maximum number of bytes that can be waiting before an inter IPU copy
    between IPUs is scheduled.
    """
    self.maximum_inter_ipu_copies_buffer_size = 0
    """
    The maximum number of bytes that can be waiting before a cluster of
    send/recv instructions to/from the host is scheduled. These are lowered to
    stream copies that can be merged by Poplar.
    """
    self.maximum_send_recv_cluster_size = 0
    """
    The maximum size (in bytes) a cluster of reduce operations can reach
    before it is scheduled. These clusters are lowered to popops ReduceMany
    operations.
    """
    self.maximum_reduce_many_buffer_size = 0
    """
    The maximum size (in bytes) a cluster of all gather operations can reach
    before it is scheduled. These clusters are lowered to popops AllGather
    operations.
    """
    self.maximum_all_gather_buffer_size = 0
    """
    The minimum size (in bytes) a tensor must be in order to be considered for
    being stored in remote memory.
    """
    self.minimum_remote_tensor_size = 128
    """
    Whether to merge compatible remote buffers. Merging of remote buffers can
    allow for more code re-use if the only difference between computations are
    the remote buffers being accessed. Must be a
    :py:class:`~tensorflow.python.ipu.config.MergeRemoteBuffersBehaviour`.
    """
    self.merge_remote_buffers = MergeRemoteBuffersBehaviour.IF_BENEFICIAL
    """
    If True (default), more aggressive optimizations will be done on embedding
    lookups.
    """
    self.enable_gather_simplifier = True
    """
    Defines the block size for the triangular solver expander. The processing
    within each block is performed on a single tile. The control code for
    performing computations over blocks is unrolled on the device. For a matrix
    of rank ``N`` and block size ``B``, there are ``log2(N/B)`` iterations of
    the control code. The choice of this parameter therefore has to balance
    between the amount of data in a tile (lower value is better, gives better
    parallelism) and the amount of control code (larger value is better, less
    control code). A value of 0 (default) selects an implementation defined
    default.
    """
    self.triangular_solve_expander_block_size = 0
    """
    Defines the block size for the Cholesky factoriser. The processing within
    each block is performed on a single tile. The control code for performing
    computations over blocks are unrolled on the device. For a matrix of rank
    ``N`` and block size ``B``, there are ``N/B`` iterations of the control
    code. The choice of this parameter therefore has to balance between the
    amount of data in a tile (lower value is better, gives better parallelism)
    and the amount of control code (larger value is better, less control code).
    A value of 0 (default) selects an implementation defined default.
    """
    self.cholesky_block_size = 0
    """
    Enables optimizations which allow arbitrary reassociations and
    transformations of mathematical operations with no accuracy guarantees.
    Enabling this option can result in incorrect output for programs that depend
    on an exact implementation of IEEE floating point for maths functions. It
    may, however, yield faster code for programs that do not require the
    guarantees of these specifications.
    """
    self.enable_fast_math = False
    """
    Control whether or not we replace dynamicSlice/Update with
    multiSlice/Update. This can increase parallelism and provide better
    memory usage since multiSlice/Update can be planned.
    """
    self.enable_dynamic_slice_replacement = True

  def _to_protobuf(self, pb):
    super()._to_protobuf(pb)  # pylint: disable=protected-access

    pb.prefetch_data_streams = self.prefetch_data_streams
    pb.enable_multi_slice_combiner = self.combine_embedding_lookups
    pb.enable_matmul_combiner = self.combine_matmuls
    pb.speed_size_config.disable_graph_outlining = not \
        self.enable_graph_outlining
    pb.speed_size_config.merge_infeed_io_copies = self.merge_infeed_io_copies
    pb.max_cross_replica_sum_buffer_size = \
        self.maximum_cross_replica_sum_buffer_size
    pb.max_reduce_scatter_buffer_size = self.maximum_reduce_scatter_buffer_size
    pb.max_reduce_many_buffer_size = self.maximum_reduce_many_buffer_size
    pb.max_all_gather_buffer_size = self.maximum_all_gather_buffer_size
    pb.max_inter_ipu_copies_buffer_size = \
        self.maximum_inter_ipu_copies_buffer_size
    pb.max_send_recv_cluster_size = self.maximum_send_recv_cluster_size
    pb.minimum_remote_tensor_size = self.minimum_remote_tensor_size
    pb.remote_buffer_merging_mode = self.merge_remote_buffers.value
    pb.disable_gather_simplifier = not self.enable_gather_simplifier
    pb.triangular_solve_expander_block_size = \
        self.triangular_solve_expander_block_size
    pb.cholesky_block_size = self.cholesky_block_size
    pb.enable_fast_math = self.enable_fast_math
    pb.enable_dynamic_slice_replacement = \
      self.enable_dynamic_slice_replacement


class _SchedulingConfig(_ConfigBase):
  def __init__(self):
    """
    A :py:class:`~tensorflow.python.ipu.config.SchedulingAlgorithm`.
    If `SchedulingAlgorithm.CHOOSE_BEST` (default), several schedules will be
    created and the one with the lowest predicted liveness chosen.
    Setting this to a specific scheduling algorithm forces the compiler to use
    that algorithm when ordering the instructions.
    """
    self.algorithm = SchedulingAlgorithm.CHOOSE_BEST
    """
    Controls how far the ``LOOK_AHEAD`` scheduling algorithm can look beyond a
    given scheduling decision to understand the max-liveness implications. This
    search space grows very quickly and can take an unacceptable amount of time
    for large values. Only for `SchedulingAlgorithm.LOOK_AHEAD`.
    """
    self.maximum_scheduler_lookahead_depth = 5
    """
    The upper-limit to the size of the ``LOOK_AHEAD`` scheduling algorithm's
    search space to guarantee that it will terminate in a reasonable amount of
    time. Only for `SchedulingAlgorithm.LOOK_AHEAD`.
    """
    self.maximum_scheduler_search_space_size = 64

  def _to_protobuf(self, pb):
    pb.speed_size_config.scheduler_selection = self.algorithm.value
    pb.max_scheduler_lookahead_depth = self.maximum_scheduler_lookahead_depth
    pb.max_scheduler_search_space_size = \
        self.maximum_scheduler_search_space_size


class IPUConfig(_ConfigBase):
  """
  A nested Python structure containing all IPU configuration options, organized
  into sub-categories.
  This docstring is overwritten when an IPUConfig instance is created.
  """
  def __init__(self):
    """
    Whether or not to recompute instructions during training.
    If this is enabled then we will attempt to pattern match
    instructions/pipeline stages in the forward pass and recompute them in the
    backward pass to avoid having to preserve activations which increase the
    maximum memory liveness. Enabling this option can reduce memory usage at
    the expense of extra computation. Stateful operations cannot be recomputed.
    """
    self.allow_recompute = False
    """
    The order in which IPUs are selected and mapped to physical IPU devices when
    using multi-IPU devices. Must be one of
    :py:class:`~tensorflow.python.ipu.config.SelectionOrder`.
    """
    self.selection_order = SelectionOrder.AUTO
    """
    Specifies the directory in which serialized Poplar executables will be
    saved. The value must be a valid path. The default ("") disables executable
    serialization.
    """
    self.serialization_output_folder = ""
    """
    Set the Poplar compilation options for the session. Must be a
    dictionary of valid Poplar compilation flags. See the `Engine` class in the
    Poplar API reference for the full list of options.
    """
    self.compilation_poplar_options = {}
    """
    Set the IPU options for the Graphcore Communication Library. Must be a
    dictionary of valid GCL options. See the `allReduce` function in the GCL API
    reference for the full list of options. The options will be applied to all
    applicable GCL collective operations in the graph during compilation.
    """
    self.gcl_poplar_options = {}
    """
    Configure the IPUs to be used by the session.
    The configuration describes a system consisting of multiple TensorFlow
    devices, each with control of one of more IPUs. The devices will be labeled
    ``/device:IPU:0``, ``/device:IPU:1`` and so on.

    Each device can control a specific number of IPUs, given by the ``num_ipus``
    parameter. The system will automatically select IPU configurations from the
    available IPUs, where they match the desired number of IPUs.

    Examples:

    .. code-block:: python

      config = IPUConfig()

      # Create a single TensorFlow device, with one IPU
      config.auto_select_ipus = 1

      # Create two TensorFlow devices, with two IPUs per device.
      config.auto_select_ipus = [2, 2]

      # Create two TensorFlow devices, with one IPU in the first device and two
      # IPUs in the second device.
      config.auto_select_ipus = [1, 2]
    """
    self.auto_select_ipus: typing.Union[int, typing.List[int], typing.
                                        Tuple[int, ...]] = []
    """
    Configure the IPUs to be used by the session.

    The configuration describes a system consisting of multiple TensorFlow
    devices, each with control of one of more IPUs. The TensorFlow devices will
    be labeled ``/device:IPU:0``, ``/device:IPU:1`` and so on.

    Each TensorFlow device uses a specific configuration consisting of one or
    more IPUs from the list of devices.  These can be found by running the
    Graphcore utility ``gc-info -l``.  For instance, the following listing shows
    the device configurations available on a system with 16 IPUs.

    .. code-block:: shell

        user@host:~$ gc-info -l
        Graphcore device listing:

        -+- Id:  [0], type:      [PCIe], PCI Domain: [0000:1a:00.0]
        -+- Id:  [1], type:      [PCIe], PCI Domain: [0000:1b:00.0]
        -+- Id:  [2], type:      [PCIe], PCI Domain: [0000:23:00.0]
        -+- Id:  [3], type:      [PCIe], PCI Domain: [0000:24:00.0]
        -+- Id:  [4], type:      [PCIe], PCI Domain: [0000:3d:00.0]
        -+- Id:  [5], type:      [PCIe], PCI Domain: [0000:3e:00.0]
        -+- Id:  [6], type:      [PCIe], PCI Domain: [0000:43:00.0]
        -+- Id:  [7], type:      [PCIe], PCI Domain: [0000:44:00.0]
        -+- Id:  [8], type:      [PCIe], PCI Domain: [0000:8b:00.0]
        -+- Id:  [9], type:      [PCIe], PCI Domain: [0000:8c:00.0]
        -+- Id: [10], type:      [PCIe], PCI Domain: [0000:8e:00.0]
        -+- Id: [11], type:      [PCIe], PCI Domain: [0000:8f:00.0]
        -+- Id: [12], type:      [PCIe], PCI Domain: [0000:b8:00.0]
        -+- Id: [13], type:      [PCIe], PCI Domain: [0000:b9:00.0]
        -+- Id: [14], type:      [PCIe], PCI Domain: [0000:ba:00.0]
        -+- Id: [15], type:      [PCIe], PCI Domain: [0000:bb:00.0]
        -+- Id: [16], type: [Multi IPU]
        |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
        |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
        -+- Id: [17], type: [Multi IPU]
        |--- PCIe Id:  [4], DNC Id: [0], PCI Domain: [0000:3d:00.0]
        |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:43:00.0]
        -+- Id: [18], type: [Multi IPU]
        |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:24:00.0]
        |--- PCIe Id:  [1], DNC Id: [1], PCI Domain: [0000:1b:00.0]
        -+- Id: [19], type: [Multi IPU]
        |--- PCIe Id:  [2], DNC Id: [0], PCI Domain: [0000:23:00.0]
        |--- PCIe Id:  [0], DNC Id: [1], PCI Domain: [0000:1a:00.0]
        -+- Id: [20], type: [Multi IPU]
        |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
        |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
        -+- Id: [21], type: [Multi IPU]
        |--- PCIe Id: [12], DNC Id: [0], PCI Domain: [0000:b8:00.0]
        |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:ba:00.0]
        -+- Id: [22], type: [Multi IPU]
        |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:8c:00.0]
        |--- PCIe Id: [11], DNC Id: [1], PCI Domain: [0000:8f:00.0]
        -+- Id: [23], type: [Multi IPU]
        |--- PCIe Id: [10], DNC Id: [0], PCI Domain: [0000:8e:00.0]
        |--- PCIe Id:  [8], DNC Id: [1], PCI Domain: [0000:8b:00.0]
        -+- Id: [24], type: [Multi IPU]
        |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
        |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
        |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
        |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
        -+- Id: [25], type: [Multi IPU]
        |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:24:00.0]
        |--- PCIe Id:  [1], DNC Id: [1], PCI Domain: [0000:1b:00.0]
        |--- PCIe Id:  [2], DNC Id: [2], PCI Domain: [0000:23:00.0]
        |--- PCIe Id:  [0], DNC Id: [3], PCI Domain: [0000:1a:00.0]
        -+- Id: [26], type: [Multi IPU]
        |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
        |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
        |--- PCIe Id: [12], DNC Id: [2], PCI Domain: [0000:b8:00.0]
        |--- PCIe Id: [14], DNC Id: [3], PCI Domain: [0000:ba:00.0]
        -+- Id: [27], type: [Multi IPU]
        |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:8c:00.0]
        |--- PCIe Id: [11], DNC Id: [1], PCI Domain: [0000:8f:00.0]
        |--- PCIe Id: [10], DNC Id: [2], PCI Domain: [0000:8e:00.0]
        |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:8b:00.0]
        -+- Id: [28], type: [Multi IPU]
        |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
        |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
        |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
        |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
        |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:24:00.0]
        |--- PCIe Id:  [1], DNC Id: [5], PCI Domain: [0000:1b:00.0]
        |--- PCIe Id:  [2], DNC Id: [6], PCI Domain: [0000:23:00.0]
        |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
        -+- Id: [29], type: [Multi IPU]
        |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:b9:00.0]
        |--- PCIe Id: [15], DNC Id: [1], PCI Domain: [0000:bb:00.0]
        |--- PCIe Id: [12], DNC Id: [2], PCI Domain: [0000:b8:00.0]
        |--- PCIe Id: [14], DNC Id: [3], PCI Domain: [0000:ba:00.0]
        |--- PCIe Id:  [9], DNC Id: [4], PCI Domain: [0000:8c:00.0]
        |--- PCIe Id: [11], DNC Id: [5], PCI Domain: [0000:8f:00.0]
        |--- PCIe Id: [10], DNC Id: [6], PCI Domain: [0000:8e:00.0]
        |--- PCIe Id:  [8], DNC Id: [7], PCI Domain: [0000:8b:00.0]
        -+- Id: [30], type: [Multi IPU]
        |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:3e:00.0]
        |--- PCIe Id:  [7], DNC Id: [1], PCI Domain: [0000:44:00.0]
        |--- PCIe Id:  [4], DNC Id: [2], PCI Domain: [0000:3d:00.0]
        |--- PCIe Id:  [6], DNC Id: [3], PCI Domain: [0000:43:00.0]
        |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:24:00.0]
        |--- PCIe Id:  [1], DNC Id: [5], PCI Domain: [0000:1b:00.0]
        |--- PCIe Id:  [2], DNC Id: [6], PCI Domain: [0000:23:00.0]
        |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
        |--- PCIe Id: [13], DNC Id: [8], PCI Domain: [0000:b9:00.0]
        |--- PCIe Id: [15], DNC Id: [9], PCI Domain: [0000:bb:00.0]
        |--- PCIe Id: [12], DNC Id: [10], PCI Domain: [0000:b8:00.0]
        |--- PCIe Id: [14], DNC Id: [11], PCI Domain: [0000:ba:00.0]
        |--- PCIe Id:  [9], DNC Id: [12], PCI Domain: [0000:8c:00.0]
        |--- PCIe Id: [11], DNC Id: [13], PCI Domain: [0000:8f:00.0]
        |--- PCIe Id: [10], DNC Id: [14], PCI Domain: [0000:8e:00.0]
        |--- PCIe Id:  [8], DNC Id: [15], PCI Domain: [0000:8b:00.0]

    Examples based on the listing above:

    .. code-block:: python

        config = IPUConfig()

        # Create a single TensorFlow device with 1 IPU at PCI address
        # 0000:1a:00.0 by using IPU configuration index 0
        config.select_ipus = 0

        # Create a single TensorFlow device with 1 IPU at PCI address
        # 0000:8b:00.0 by using IPU configuration index 8
        config.select_ipus = 8

        # Create two TensorFlow devices, with one IPU each, being devices at
        # indices 0 and 1
        config.select_ipus = [0, 1]

        # Create two TensorFlow devices, with four IPUs each. The device
        # configurations at indices 24 (0000:3e:00.0, 0000:44:00.0,
        # 0000:3d:00.0, 000:43:00.0) and 25 (0000:24:00.0, 0000:1b:00.0,
        # 0000:23:00.0, 00:1a:00.0)
        config.select_ipus = [24, 25]

        # Create four TensorFlow devices each with one IPU, at addresses
        # 0000:1a:00.0, 0000:1b:00.0, 0000:23:00.0, 0000:24:00.0.
        config.select_ipus = [0, 1, 2, 3]
    """
    self.select_ipus: typing.Union[int, typing.List[int], typing.
                                   Tuple[int, ...]] = []
    """
    Sub-category containing configuration options that affect convolutions.
    """
    self.convolutions = _ConvolutionConfig()
    """
    Sub-category containing configuration options to control when to attach to
    IPU devices.
    """
    self.device_connection = _IPUDeviceConnectionConfig()
    """
    Sub-category containing configuration options that affect slice operations.
    """
    self.slices = _SliceConfig()
    """
    Sub-category containing experimental configuration options that may be
    changed or removed with short or no notice.
    """
    self.experimental = _ExperimentalConfig()
    """
    Sub-category containing configuration options that affect the floating point
    behaviour of the IPU devices, including stochastic rounding and behaviour
    when an overflow is encountered during execution. For more information,
    see :ref:`controlling-half-unit`.
    """
    self.floating_point_behaviour = _FloatingPointBehaviourConfig()
    """
    Sub-category containing configuration options that affect parallel I/O on
    a subset of tiles. For more information, see :ref:`i-o-tiles`.
    """
    self.io_tiles = _IOTilesConfig()
    """
    Sub-category containing configuration options related to the IPU model. Note
    that these will only have an effect if you are running with the IPU model
    enabled. For more information, see :ref:`env-var-section`.
    """
    self.ipu_model = _IPUModelConfig()
    """
    Sub-category containing configuration options that affect matmuls.
    """
    self.matmuls = _MatmulConfig()
    """
    Sub-category containing configuration options that affect normalizations.
    Note that these options will be applied to all normalisation operations
    encountered (Fused Batch Norm, IPU Specific Group Norm, IPU Specific Layer
    Norm and IPU Specific Instance Norm).
    """
    self.norms = _NormConfig()
    """
    Sub-category containing configuration options that control a variety of
    optimizations made when lowering the TensorFlow graph to Poplar.
    """
    self.optimizations = _OptimizationConfig()
    """
    Sub-category containing configuration options that affect pooling
    operations.
    """
    self.pooling = _PoolingConfig()
    """
    Sub-category containing configuration options that affect the scheduling of
    operations in the graph during compilation.
    """
    self.scheduling = _SchedulingConfig()
    # This only needs to be called in this base config, not nested configs. It
    # generates the docstring of this class and propagates deprecation.
    self._finalize_base_config()  # pylint: disable=protected-access

  def _create_protobuf(self):
    """ Create an IpuOptions protobuf from this IPUConfig """
    pb = IpuOptions()
    self._to_protobuf(pb)  # pylint: disable=protected-access
    return pb

  def _to_protobuf(self, pb):
    # Only one of (auto_)select_ipus can be set.
    if (self.auto_select_ipus and self.select_ipus):
      raise Exception(
          "Only one of `auto_select_ipus` and `select_ipus` can be set.")
    if isinstance(self.auto_select_ipus, int):
      self.auto_select_ipus = [self.auto_select_ipus]
    if isinstance(self.select_ipus, int):
      self.select_ipus = [self.select_ipus]
    if len(set(self.select_ipus)) != len(self.select_ipus):
      raise ValueError("All MultiIPU indices in `select_ipus` must be unique.")
    if self.select_ipus and running_on_ipu_model():
      raise Exception("When using the IPUModel, the devices to attach to can"
                      " only be specified with `auto_select_ipus`.")

    # distributed_batch_norm_replica_group_size is not supported with I/O tiles.
    if self.io_tiles.num_io_tiles > 0 and \
        self.norms.experimental.distributed_batch_norm_replica_group_size > 1:
      raise ValueError(
          "norms.experimental.distributed_batch_norm_replica_group_size is not"
          " supported with I/O tiles.")

    pb.creator_id = config_pb2.IpuOptionsCreator.IPU_UTILS

    pb.speed_size_config.allow_recompute = self.allow_recompute
    pb.selection_order = self.selection_order.value
    pb.serialization_folder = self.serialization_output_folder
    _poplar_options_to_protobuf(self.compilation_poplar_options,
                                pb.compilation_options)
    _poplar_options_to_protobuf(self.gcl_poplar_options, pb.gcl_options)

    for device_index in self.select_ipus:
      dev = pb.device_config.add()
      dev.cfg_index = device_index
    for device_count in self.auto_select_ipus:
      dev = pb.device_config.add()
      dev.auto_count = device_count

    # Go deeper into nested configs.
    super()._to_protobuf(pb)  # pylint: disable=protected-access

  def configure_ipu_system(self, device="cpu"):
    """
    Configure the IPU system with this config.

    Args:
      device: The CPU device which is local to the IPU hardware.
    """
    configure_ipu_system(self, device)


# pylint: enable=pointless-string-statement

# Sphinx imports the IPUConfig to inspect its docstring in order to generate
# documentation, but it doesn't instantiate it. We need to instantiate it once
# here to generate the class docstring from the config structure.
IPUConfig()


def configure_ipu_system(config, device="cpu", reset_configuration=True):
  """Configure an IPU system with an IPUConfig or IpuOptions instance.

  Args:
    config: An IPUConfig instance or IpuOptions configuration protobuf.
    device: The TensorFlow virtual CPU device which is local to the IPU
            hardware.
    reset_configuration: Whether to reset any existing IPU configurations.

  Returns:
    None
  """
  if isinstance(config, IPUConfig):
    config = config._create_protobuf()  # pylint: disable=protected-access

  if not isinstance(config, config_pb2.IpuOptions):
    raise TypeError("`config` must be an IpuOptions instance.")

  try:
    existing_configs = get_ipu_config()
  except RuntimeError:
    reset_configuration = False
  else:
    # get_ipu_config returns the same config for all executors.
    config_changed = existing_configs[0] != config
    reset_configuration &= config_changed

  if reset_configuration:
    logging.warn(
        "Resetting existing IPU configuration before applying new configuration"
    )
    reset_ipu_configuration()

  g = ops.Graph()
  with g.as_default():
    with ops.device(device):
      cfg_op = gen_ipu_ops.ipu_configure_hardware(config.SerializeToString())

  with session_lib.Session(graph=g) as sess:
    sess.run(cfg_op)


def reset_ipu_configuration():
  """ Reset the IPU configuration in preparation for it to be reconfigured.
  Blocks until all currently configured IPU devices have finished executing.

  Note that this function does not currently support reseting IPUs that are
  running in parallel python threads.
  """
  sync_ops = []

  try:
    configs = get_ipu_config()
  except RuntimeError:
    # No devices have been configured.
    return

  g = ops.Graph()
  with g.as_default():
    # get_ipu_config returns 1 config per executor
    for i in range(len(configs)):
      device_name = f"/device:IPU:{i}"
      with ops.device(device_name):
        sync_ops.append(gen_poputil_ops.device_sync())

    with ops.device("CPU"):
      with ops.control_dependencies(sync_ops):
        # Wait for sync to complete before clearing.
        sync_ops.append(gen_ipu_ops.ipu_reset_devices())
        sync_ops.append(gen_ipu_ops.ipu_clear_all_xla_compilation_caches())

  with session_lib.Session(graph=g) as sess:
    sess.run(sync_ops)


def get_ipu_config(session=None):
  """Get the configuration of an IPU system.

  Args:
    session: An optional session on which to execute.

  Returns:
    A list of IpuOption instances, one for each PoplarExecutor.
  """
  configurations = None

  # Get the serialized output.
  if executing_eagerly():
    assert not session, "No session is required for eager execution."
    configurations = gen_ipu_ops.ipu_get_configuration().numpy()
  else:
    if session:
      configurations = session.run(gen_ipu_ops.ipu_get_configuration())
    else:
      g = ops.Graph()
      with g.as_default():
        with ops.device("CPU"):
          with session_lib.Session(graph=g) as s:
            configurations = s.run(gen_ipu_ops.ipu_get_configuration())

  # Deserialize and determine if a valid config exists,
  # i.e. user has successfully called ipu_configure_hardware.
  deserialized = []
  valid = False
  for conf in configurations:
    # Deserialize.
    opt = IpuOptions()
    opt.ParseFromString(conf)
    deserialized.append(opt)

    valid |= len(opt.device_config) > 0

  if not valid:
    raise RuntimeError("No IPU devices configured.")

  return deserialized

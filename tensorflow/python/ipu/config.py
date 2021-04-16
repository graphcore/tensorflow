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
IPU configuration classes and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import ast
import collections
import difflib
import inspect
import pydoc

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.python.platform import tf_logging as logging


def annotation_to_str(node):
  """
  Construct a type hint string from an AST annotation.
  """
  if isinstance(node, str):
    return node
  if isinstance(node, ast.Str):
    return node.s
  if isinstance(node, ast.Attribute):
    return f"{annotation_to_str(node.value)}.{annotation_to_str(node.attr)}"
  if isinstance(node, ast.Subscript):
    return f"{annotation_to_str(node.value)}[{annotation_to_str(node.slice)}]"
  if isinstance(node, ast.Slice):

    def helper(v):
      return annotation_to_str(getattr(node, v, ast.Str("")))

    return ":".join(map(helper, ['lower', 'upper', 'step']))
  if isinstance(node, ast.Index):
    return annotation_to_str(node.value)
  if isinstance(node, ast.Tuple):
    return ', '.join(map(annotation_to_str, node.elts))
  if isinstance(node, ast.Ellipsis):
    return "..."
  if isinstance(node, ast.Name):
    return node.id
  raise Exception(f"Unhandled {node} when converting type hint to string.")


def get_type_check_fn_from_AST_type_hints(node):
  """
  Function that parses the type hints in an AST AnnAssign node and converts them
  into a callable that checks the type of a value `v` against the type hints.
  Covers basic usage of typing.List, typing.Tuple and typing.Union according to
  the following formal grammar:
  L -> typing.List
  T -> typing.Tuple
  U -> typing.Union
  t -> int | list | tuple | str | float | any python built-in type that can
       be located by pydoc.locate() | any symbol that is available in globals()
       when the returned function is called.

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

  Essentially this function allows you to enforce simple type hints.
  """
  assert isinstance(node, ast.AnnAssign), "Only AnnAssign AST nodes allowed."

  def check_type(typ):
    # Strict type comparison over isinstance.
    return lambda v: type(v) == typ  # pylint: disable=unidiomatic-typecheck

  def is_typing_module_attr(node):
    return isinstance(node, ast.Attribute) and \
        isinstance(node.value, ast.Name) and node.value.id == "typing"

  def helper(node):
    # Handle "typing.List", "typing.Tuple"
    if is_typing_module_attr(node):
      if node.attr == "List":
        return check_type(list)
      if node.attr == "Tuple":
        return check_type(tuple)
      raise Exception(f"Unsupported 'typing' attribute {node.attr}")

    # Handle any arbitrary built-in e.g. 'int', or class/enum names.
    if isinstance(node, ast.Name):
      # Search for the variable in globals() at the time of execution. Failing
      # that, use pydoc to locate it.
      return lambda v: type(v) == globals().get(node.id, pydoc.locate(node.id))  # pylint: disable=unidiomatic-typecheck

    # Subscripts, e.g. X[...]
    if isinstance(node, ast.Subscript):
      lhs = node.value
      if is_typing_module_attr(lhs):
        # e.g. Union[int, str], check v for all union types
        if lhs.attr == "Union":
          assert isinstance(node.slice, ast.Index)
          assert isinstance(node.slice.value, ast.Tuple)
          type_fns = [helper(n) for n in node.slice.value.elts]
          return lambda v: any(fn(v) for fn in type_fns)
        # e.g. Tuple[int, ...], check v elementwise for each type
        if lhs.attr == "Tuple":
          check_tuple = check_type(tuple)
          # single element Tuple: check the single element in v for type
          if isinstance(node.slice.value, ast.Name):
            type_fn = helper(node.slice.value)
            return lambda v: check_tuple(v) and len(v) == 1 and type_fn(v[0])
          # more than one element Tuple
          if isinstance(node.slice.value, ast.Tuple):
            # e.g. Tuple[int, ...], check each element in v for the same type
            if len(node.slice.value.elts) > 1 and isinstance(
                node.slice.value.elts[1], ast.Ellipsis):
              type_fn = helper(node.slice.value.elts[0])
              return lambda v: check_tuple(v) and all([type_fn(e) for e in v])
            # e.g. Tuple[int, str], pair-wise (element, type) check
            type_fns = [helper(n) for n in node.slice.value.elts]
            return lambda v: check_tuple(v) and len(v) == len(
                type_fns) and all([fn(e) for fn, e in zip(type_fns, v)])
        # e.g. List[int], check each element in v for the same type
        if lhs.attr == "List":
          assert not isinstance(
              node.slice.value,
              ast.Tuple), "List with more than one type not allowed."
          check_list = check_type(list)
          type_fn = helper(node.slice.value)
          return lambda v: check_list(v) and all([type_fn(e) for e in v])
        raise Exception(f"Unsupported 'typing' attribute {lhs.attr}")
      raise Exception(
          f"Only 'typing' module types can be subscripted in hints. {lhs}")
    raise Exception(f"Unhandled AST node type in type hint {node}")

  return helper(node.annotation)


def called_from_instance_init(instance, calling_frame):
  """ Helper to check a calling_frame is in an instance's __init__ """
  from_init = inspect.getframeinfo(calling_frame).function == "__init__"
  from_this = calling_frame.f_locals.get('self') is instance
  return from_init and from_this


def build_full_attribute_name_from_call_stack(init_frame):
  """
  Given a frame in a call stack that points to an assignment which is inside
  a config structure, traverse up the structure by following the assignments to
  build the full name of the attribute relative to the base of the config
  structure. For example, given the following config structure:
  ```
  class NestedConfig(ConfigBase):
    def __init__(self):
      self.attribute = 1

  class ExampleConfig(ConfigBase):
    def __init__(self):
      nested_config = NestedConfig(ConfigBase)
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
    assignment_node = get_assignment_node_from_call_frame(cur_frame)
    if isinstance(assignment_node, ast.AnnAssign):
      name = assignment_node.target.attr
    else:
      name = assignment_node.targets[0].attr

    name_parts = [name] + name_parts

    # Finish if we reached the root of the nested structure.
    parent_class = cur_frame.f_back.f_locals.get('self', None)
    if not isinstance(parent_class, ConfigBase):
      break
    # Go up the assignment stack until we reach the config root.
    cur_frame = cur_frame.f_back
  return ".".join(name_parts), len(name_parts)


def get_docstring_above_calling_line(call_frame):
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
  # Find the docstring by looking for the AST node for the line above the
  # assigning statement. We have to use the AST node since the docstring
  # could be over a number of source lines.
  assignment_file = call_frame.f_code.co_filename
  source_index = get_source_index(assignment_file)
  nodes = source_index.get(call_frame.f_lineno - 1, [])
  # (Docstrings are Exprs with a Str in them in AST)
  if len(nodes) == 2 and isinstance(nodes[0], ast.Expr) and isinstance(
      nodes[0].value, ast.Str):
    return nodes[0].value.s
  return "No description provided."


def get_assignment_type_and_checker(assign_node, rhs):
  """
  Given an AST assignment node, get the type of the RHS of the assignment as
  a string and also build a function that will check a value against the type of
  the RHS. If the assignment node is not annotated, we simply use the type of v.

  Args:
    assign_node: The AST node for the assignment.
    rhs: The right hand side (target) of the assignment as a Python value.
  """
  # Find possible types...
  if isinstance(assign_node, ast.AnnAssign):
    # ...from Python's type hint annotations
    check_type_fn = get_type_check_fn_from_AST_type_hints(assign_node)
    # Reconstruct the type hint string from the AST node.
    attr_type = annotation_to_str(assign_node.annotation)
  else:
    # ...from the initial value
    check_type_fn = lambda value: type(value) == type(rhs)  # pylint: disable=unidiomatic-typecheck
    attr_type = type(rhs).__name__
  return attr_type, check_type_fn


_DEPRECATIONS = {}


def deprecate_config_attribute(name, msg):
  """
  Class decorator to deprecate an attribute in a nested ConfigBase structure.
  Stores the deprecation in the class's DEPRECATIONS attribute so it can be
  used later when we determine attribute metadata on initialization.

  Args:
    name: The name of the attribute on the ConfigBase this decorates to
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


_FILENAME_SOURCE_INDEXES = {}


def get_source_index(filename):
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


def get_assignment_node_from_call_frame(frame):
  """
  Helper to get the Assign or AnnAssign AST node for a call frame.
  The call frame will point to a specific file and line number, and we use the
  source index to retrieve the AST nodes for that line.
  """
  filename = frame.f_code.co_filename
  # Go up the AST from a node in the call frame line until we find an Assign or
  # AnnAssign, since the (Ann)Assign may be over multiple lines.
  nodes_in_line = get_source_index(filename).get(frame.f_lineno, [])
  cur_node = nodes_in_line[0]
  while cur_node:
    if isinstance(cur_node, (ast.Assign, ast.AnnAssign)):
      return cur_node
    cur_node = cur_node.parent
  raise Exception("Could not find AST assignment node in the line"
                  f" {filename}:{frame.f_lineno}")


class AttributeMetadata:
  """
  Encapsulates the metadata for an attribute in a nested ConfigBase structure.

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
  def __init__(self,
               name,
               doc="",
               depth=0,
               default=None,
               deprecated_msg=None,
               attr_type=None,
               check_type_fn=lambda v: True):
    self.name = name
    self.__doc__ = inspect.cleandoc(doc)  # Normalize docstring indentation.
    self.deprecated = deprecated_msg is not None
    self.deprecated_msg = deprecated_msg
    self.type = attr_type
    self.default = default
    self._check_type_fn = check_type_fn
    self._depth = depth

  def check_type(self, value):
    """
    Checks if `value` is one of the allowed types for this option. Throws a
    TypeError if not.
    """
    if not self._check_type_fn(value):
      raise TypeError(
          f"Trying to set {self.name} to {value}, but it must be of"
          f" type {self.type}")

  def warn_if_deprecated(self):
    """
    Outputs a warning if this option/category is deprecated.
    """
    if self.deprecated:
      logging.warn(f"{self.name} has been deprecated: {self.deprecated_msg}")

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
      lines.append("   .. note::")
      lines.append(f"      DEPRECATED: {self.deprecated_msg}")

    # Indent and add the docstring.
    for line in self.__doc__.split('\n'):
      lines.append("   " + line)

    # Indent the entire block by the depth.
    return '\n'.join(["   " * (self._depth + 1) + l for l in lines]) + '\n\n'


class ConfigBase(object):
  """
  A class that can be used to create a user-friendly hierarchical structure of
  attributes that can be converted into a protobuf for the Poplar XLA backend.
  Non-root classes in the structure are hidden from the user so all attributes
  are accessed with chained dot notation from the root class.

  To use, create a root ConfigBase with some (typed) attributes:
  ```
  class Config(ConfigBase):
    def __init__(self):
      self.option1: int = 1
      self.option2: typing.Union[int, str] = 2
      self._finalize_base_config()
  ```

  The root can then have ConfigBase attributes itself, building the hierarchy:
  ```
  class _HiddenCategoryClass(ConfigBase):
    def __init__(self):
      self.option3 = 3

  class Config(ConfigBase):
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
  Since nested ConfigBase attributes are also attributes, they can also be
  given docstrings to e.g. describe those general "categories".


  Types
  ~~~~~
  Basic type hints are supported for non-category attributes. If there are none,
  the type of the initial value will be used. Types are strictly enforced when
  a user sets any attribute. For a full list of supported type hints, see the
  formal grammar in `get_type_check_fn_from_AST_type_hints`.


  Some examples of supported type hints:
  ```
  class _B(ConfigBase):
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
  class _B(ConfigBase):
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

  Sphinx documentation is automatically generated for an entire ConfigBase
  config from the structure and the docstrings. The nested ConfigBase *classes*
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
  ConfigBase._finalize_base_config is called at the end of the base config
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
    full_name, depth = build_full_attribute_name_from_call_stack(caller_frame)

    # Find the docstring above the assignment statement.
    docstring = get_docstring_above_calling_line(caller_frame)

    # Find deprecation through earlier registration on _DEPRECATIONS.
    deprecated_msg = _DEPRECATIONS.get((self.__class__, k), None)

    # Find type string and type checking function from AST node.
    assign_node = get_assignment_node_from_call_frame(caller_frame)
    attr_type, type_checker = get_assignment_type_and_checker(assign_node, v)

    # Hide default and types for categories.
    default = v
    if isinstance(v, ConfigBase):
      # Keep track of nested configs for convenience.
      self._nested_configs.append(k)
      default = None
      attr_type = None

    self._user_attributes[k] = AttributeMetadata(full_name,
                                                 doc=docstring,
                                                 depth=depth,
                                                 default=default,
                                                 deprecated_msg=deprecated_msg,
                                                 attr_type=attr_type,
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
      if not called_from_instance_init(self, caller_frame):
        # Call the failing getter for a suggestion and exception.
        self.__getattr__(k)

    # Adding a new user-facing attribute.
    self._create_new_user_attribute(k, v, caller_frame)

  def _to_protobuf(self, pb):
    """
    Convert nested ConfigBases to a protobuf.
    If an inheritor wants to modify the protobuf, it must implement this method.
    If it also contains nested configs, it should call this method too to
    recurse into the nested configs to allow them to modify the protobuf too.
    """
    for config_name in self._nested_configs:
      getattr(self, config_name)._to_protobuf(pb)  # pylint: disable=protected-access

  def _create_protobuf(self):
    """ Create an IpuOptions protobuf from this ConfigBase """
    pb = IpuOptions()
    self._to_protobuf(pb)  # pylint: disable=protected-access
    return pb

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
        metadata.deprecated = _parent_metadata.deprecated
        metadata.deprecated_msg = _parent_metadata.deprecated_msg

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
            metadata for. Must be its full name relative to the config
            this method is being called on.
    Returns:
      An `_AttributeMetadata` object containing the metadata for the attribute.
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

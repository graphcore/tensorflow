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
# ==============================================================================
import copy
import enum
import json
import os
import typing

from absl.testing import parameterized
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.compiler.plugin.poplar.driver.config_pb2 import \
    IpuDeviceConnectionType, IpuSelectionOrder, IpuExecutionProfileType, \
    IpuSchedulingAlgorithm
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

disable_v2_behavior()


# pylint: disable=pointless-string-statement
# pylint: disable=protected-access
class TestConfigNested4(ipu.config._ConfigBase):
  def __init__(self):
    self.attr4 = False
    self.attr7: TestEnum = TestEnum.APPLE


class TestConfigNested3(ipu.config._ConfigBase):
  def __init__(self):
    """
    This is the docstring for attr3
    """
    self.attr3 = "test"
    self.attr0 = 1

  def _to_protobuf(self, pb):
    pb['nested3'] = [self.attr3, self.attr0]
    super()._to_protobuf(pb)


@ipu.config.deprecate_config_attributes(
    {"attr5": "attr5 is no longer relevant."})
class TestConfigNested2(ipu.config._ConfigBase):
  def __init__(self):
    self.attr2 = 1
    self.nested4 = TestConfigNested4()
    """
    This is a deprecated attribute, but it's under a config that's also
    deprecated. When the config's deprecation is propagated, it shouldn't
    overwrite this attribute's deprecation message.
    """
    self.attr5: int = 1

  def _to_protobuf(self, pb):
    pb['nested2'] = [self.attr2]
    super()._to_protobuf(pb)


class TestEnum(enum.Enum):
  APPLE = 1
  ORANGE = 2
  BANANA = 3
  PEAR = 5


@ipu.config.deprecate_config_attribute(
    "attr1", "Attribute has been removed as it doesn't have an effect.")
@ipu.config.deprecate_config_attribute("nested2",
                                       "Category is no longer relevant.")
class TestConfigNested1(ipu.config._ConfigBase):
  # This nested config doesn't call _to_protobuf, but its children should still
  # have their _to_protobuf called.
  def __init__(self):
    """
    This is the docstring for attr1
    """
    self.attr1: typing.Union[int, list] = 1
    """
    This attribute has in-line RST in its docstring which should be preserved.

    .. code-block:: python

      code
    """
    self.RST_attr = 2
    """
    This is a docstring that tests deprecating a nested config
    """
    self.nested2 = TestConfigNested2()
    self.nested3 = TestConfigNested3()


class TestConfig(ipu.config._ConfigBase):
  def __init__(self):
    """
    This is the docstring for attr0
    """
    self.attr0: typing.Union[int, str, list] = 1
    """
    This is an attribute with a very advanced type hint
    """
    self.attr6: typing.Tuple[typing.List[typing.Union[float, typing.
                                                      Tuple[str]]], ...] = ([
                                                          1.0
                                                      ],)
    """
    This is the docstring for the nested1 category
    """
    self.nested1 = TestConfigNested1()
    self._finalize_base_config()

  def _to_protobuf(self, pb):
    pb['base_config'] = [self.attr0]
    super()._to_protobuf(pb)


class ConfigBaseTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """
  Tests the basic mechanics of the ipu.config._ConfigBase class using a custom
  config.
  """
  def testGetFullName(self):
    """
    A _ConfigBase can contain arbitrarily nested _ConfigBase attributes.
    These nested classes should be invisible to the user, as they're an
    implementation detail. The user only knows about the root _ConfigBase, so
    all attribute names should be relative to this root.
    """
    test_config = TestConfig()

    self.assertEqual(test_config._get_full_name('attr0'), 'attr0')
    self.assertEqual(test_config.nested1._get_full_name('attr1'),
                     'nested1.attr1')
    self.assertEqual(test_config.nested1.nested2._get_full_name('attr2'),
                     'nested1.nested2.attr2')
    self.assertEqual(test_config.nested1.nested3._get_full_name('attr3'),
                     'nested1.nested3.attr3')
    self.assertEqual(test_config.nested1.nested3._get_full_name('attr0'),
                     'nested1.nested3.attr0')

  def testAssignment(self):
    """ Generic test that assignment works """
    test_config = TestConfig()
    test_config.attr0 = 5  # Basic
    self.assertEqual(test_config.attr0, 5)
    test_config.nested1.nested2.attr2 = 5  # Nested
    self.assertEqual(test_config.nested1.nested2.attr2, 5)

    # Nested _ConfigBase attributes should not be able to be assigned to,
    # since they're part of the config's immutable run-time structure.
    with self.assertRaisesRegex(
        Exception, "nested1.nested2 is a category and cannot be assigned to."):
      test_config.nested1.nested2 = 3
    # ...but setting one of its attrs should be fine.
    test_config.nested1.nested2.attr2 = 1

    # Assigning with a different type should fail.
    with self.assertRaisesRegex(
        TypeError,
        "Trying to set nested1.nested2.attr2 to True, but it must be of type"
        " int"):
      test_config.nested1.nested2.attr2 = True
    # Make sure the assignment didn't happen despite the error.
    self.assertEqual(test_config.nested1.nested2.attr2, 1)

    # Check we also can't assign to (non-)existent private attributes - the user
    # should only interact with public attributes.
    with self.assertRaisesRegex(
        ValueError,
        "'_private_attribute' is not a valid attribute of this config."):
      test_config._private_attribute = []
    with self.assertRaisesRegex(
        ValueError,
        "'_attr_metadata' is not a valid attribute of this config."):
      test_config._attr_metadata = {}

    # Check that attributes with the same names aren't the same buffer.
    test_config.attr0 = 5
    test_config.nested1.nested3.attr0 = 5
    self.assertEqual(test_config.attr0, test_config.nested1.nested3.attr0)
    test_config.attr0 = 3
    self.assertNotEqual(test_config.attr0, test_config.nested1.nested3.attr0)

  def testTypeHints(self):
    """ Check type hints are used to correctly check type on assignment """
    test_config = TestConfig()

    # nested1.nested2.attr5 can take an int
    test_config.nested1.nested2.attr5 = 2
    with self.assertRaisesRegex(
        TypeError, r"Trying to set .*, but it must be of type int"):
      test_config.nested1.nested2.attr5 = True
      test_config.nested1.nested2.attr5 = 2.0
      test_config.nested1.nested2.attr5 = 'test'

    # attr0 can take an int, str or list
    test_config.attr0 = 1
    test_config.attr0 = 'test'
    test_config.attr0 = [5, 'test', True]
    with self.assertRaisesRegex(
        TypeError, r"Trying to set attr0 to .*, but it must be of type"
        r" typing\.Union\[int, str, list\]"):
      test_config.attr0 = True
      test_config.attr0 = (1,)
      test_config.attr0 = 1.0

    # nested1.attr1 can be an int or a list
    test_config.nested1.attr1 = 1
    test_config.attr0 = [5, 'test', True]
    with self.assertRaisesRegex(
        TypeError,
        r"Trying to set nested1\.attr1 to .*, but it must be of type"
        r" typing\.Union\[int, list\]"):
      test_config.nested1.attr1 = True
      test_config.nested1.attr1 = (1,)
      test_config.nested1.attr1 = 1.0
      test_config.nested1.attr1 = 'test'

    # attr6 can be a tuple of lists, where each list can contain floats and
    # tuples with exactly one str in them.
    test_config.attr6 = ([1.0],)
    test_config.attr6 = ([1.0, 2.0, ('test1',)],)
    test_config.attr6 = ([5.0], [('test2',)])
    with self.assertRaisesRegex(
        TypeError, r"Trying to set attr6 to .*, but it must be of type"
        r" typing\.Tuple\[typing.List\[typing.Union\[float,"
        r" typing\.Tuple\[str\]\]\], \.\.\.\]"):
      test_config.attr6 = True
      test_config.attr6 = (True)
      test_config.attr6 = ([True],)
      test_config.attr6 = ([1.0], ['test1'])
      test_config.attr6 = ([1.0], [('test1',), 1])

  def testDidYouMean(self):
    """
    The number of attributes of a _ConfigBase class can't be modified
    outside its __init__. When an invalid attribute is asked for, suggestions
    should be made on similar valid attributes to aid usability.
    """
    test_config = TestConfig()
    # Suggestions when trying to set a non-existent nested attribute.
    with self.assertRaisesRegex(
        ValueError, "'attr5' is not a valid attribute of this config."
        " Did you mean 'nested1.attr1'?"):
      test_config.nested1.attr5  # on getting  # pylint: disable=pointless-statement
      test_config.nested1.attr5 = 3  # on setting

    # Suggestions when trying to set a non-existent nested config.
    with self.assertRaisesRegex(
        ValueError, "'nested5' is not a valid attribute of this config."
        " Did you mean 'nested1'?"):
      test_config.nested5  # pylint: disable=pointless-statement
      test_config.nested5 = 3

    # Suggestions when trying to set attribute on a deeply non-existent config.
    with self.assertRaisesRegex(
        ValueError, "'nested11' is not a valid attribute of this config."
        " Did you mean 'nested1'?"):
      test_config.nested11.nested7.nested9.nested23.attr5  # pylint: disable=pointless-statement
      test_config.nested11.nested7.nested9.nested23.attr5 = 5

    # Nothing similar shouldn't suggest anything.
    with self.assertRaisesRegex(
        ValueError,
        "'very_dissimilar_attribute' is not a valid attribute of this config."
    ):
      test_config.nested1.nested2.very_dissimilar_attribute  # pylint: disable=pointless-statement
      test_config.nested1.nested2.very_dissimilar_attribute = 3

  def testAttributeMetadata(self):
    """ Confirm that the metadata is correctly generated and accessed. """
    test_config = TestConfig()

    def check_metadata(cfg, full_name, expected_name, expected_type,
                       expected_default, expected_doc, expected_depth):
      md = cfg.get_attribute_metadata(full_name)
      self.assertEqual(md.name, expected_name)
      self.assertEqual(md.type, expected_type)
      self.assertEqual(md.default, expected_default)
      self.assertEqual(md.__doc__, expected_doc)
      self.assertEqual(md._depth, expected_depth)

    # Access base attribute metadata and confirm its contents.
    check_metadata(test_config, "attr0", "attr0",
                   "typing.Union[int, str, list]", 1,
                   "This is the docstring for attr0", 1)

    # Access attribute metadata of an attribute that's over multiple lines
    check_metadata(
        test_config, "attr6", "attr6",
        "typing.Tuple[typing.List[typing.Union[float,"
        " typing.Tuple[str]]], ...]", ([1.0],),
        "This is an attribute with a very advanced type hint", 1)

    # Try to access non-existent nested attribute.
    with self.assertRaisesRegex(
        ValueError, "Could not get attribute metadata for 'nested1.attr5': "):
      test_config.get_attribute_metadata("nested1.attr5")

    # Access nested attribute metadata and confirm its contents.
    check_metadata(test_config, "nested1.nested3.attr3",
                   "nested1.nested3.attr3", "str", "test",
                   "This is the docstring for attr3", 3)

    # Access nested attribute with same name as base attribute + no docstring
    check_metadata(test_config, "nested1.nested3.attr0",
                   "nested1.nested3.attr0", "int", 1,
                   "No description provided.", 3)

    # Access nested config metadata. Default and types should be empty.
    check_metadata(test_config, "nested1", "nested1", None, None,
                   "This is the docstring for the nested1 category", 1)

    # Access attribute metadata using a nested config
    check_metadata(test_config.nested1, "attr1", "nested1.attr1",
                   "typing.Union[int, list]", 1,
                   "This is the docstring for attr1", 2)

    # Access attribute metadata using a nested config but name isn't relative.
    with self.assertRaisesRegex(
        ValueError,
        "Could not get attribute metadata for 'nested1.attr1': 'nested1' "):
      test_config.nested1.get_attribute_metadata('nested1.attr1')

    # Trying to access attribute metadata for hidden private attributes
    # should fail too.
    with self.assertRaisesRegex(
        ValueError,
        "Could not get attribute metadata for '_attr_metadata': '_attr_"):
      test_config.get_attribute_metadata('_attr_metadata')

  def testDeprecation(self):
    """ Perform checks on deprecated attributes. """
    test_config = TestConfig()
    # Check that setting to a deprecated attribute prints a warning.
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      test_config.nested1.attr1 = 2
      self.assertRegex(
          str(mock_warn.call_args),
          "nested1.attr1 has been deprecated: Attribute has been removed as it"
          " doesn't have an effect.")
    # Check the metadata shows the attribute is deprecated.
    md = test_config.get_attribute_metadata("nested1.attr1")
    self.assertTrue(md.deprecated)
    self.assertEqual(
        md.deprecated_msg,
        "Attribute has been removed as it doesn't have an effect.")

    # Make sure an attribute that isn't deprecated and not under a deprecated
    # category doesn't print a warning and has correct metadata.
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      test_config.nested1.nested3.attr3 = "test3"
      self.assertFalse(mock_warn.called)
    md = test_config.get_attribute_metadata("nested1.nested3.attr3")
    self.assertFalse(md.deprecated)
    self.assertEqual(md.deprecated_msg, None)
    self.assertTrue("DEPRECATED: " not in md.__doc__)

    # Make sure all attributes and configs under a deprecated nested config are
    # also deprecated.
    self.assertTrue(
        test_config.get_attribute_metadata('nested1.nested2').deprecated)
    self.assertTrue(
        test_config.get_attribute_metadata('nested1.nested2.attr2').deprecated)
    self.assertTrue(
        test_config.get_attribute_metadata(
            'nested1.nested2.nested4').deprecated)
    self.assertTrue(
        test_config.get_attribute_metadata(
            'nested1.nested2.nested4.attr4').deprecated)
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      test_config.nested1.nested2.attr2 = 5
      self.assertRegex(
          str(mock_warn.call_args),
          "nested1.nested2.attr2 has been deprecated: Category is no longer"
          " relevant.")
      test_config.nested1.nested2.nested4.attr4 = True
      self.assertRegex(
          str(mock_warn.call_args),
          "nested1.nested2.nested4.attr4 has been deprecated: Category is no"
          " longer relevant.")

      # Check a deprecated attribute under a deprecated config doesn't have its
      # deprecation message overwritten.
      test_config.nested1.nested2.attr5 = 5
      self.assertRegex(
          str(mock_warn.call_args),
          "nested1.nested2.attr5 has been deprecated: attr5 is no longer"
          " relevant.")

  def testDocGeneration(self):
    """
    Make sure that the generated docstring for the base config looks like
    correct RST based on the config contents.
    """
    test_config = TestConfig()

    # All configs and attributes should have a label so they can be referenced.
    # All configs and attributes should have a .. py:attribute:: so they appear
    # as an attribute in the docs.
    # All configs and attributes should have a description.
    # All attributes should have a type and default value.
    # Deprecated configs or attributes should have a .. note:: describing why
    # they are deprecated and how to switch.
    # Configs or attributes that aren't deprecated, but are under a deprecated
    # config, should also be deprecated and contain the note of their deprecated
    # parent.
    # A nested config or attribute at a depth d should be indented by that depth
    # Options should not be repeated

    # pylint: disable=unsupported-membership-test

    # Basic attribute.
    attr0_desc = '\n'.join([
        "      .. _attr0:", "      .. py:attribute:: attr0",
        "         :type: typing.Union[int, str, list]", "         :value: 1",
        "      ", "         This is the docstring for attr0"
    ])
    self.assertTrue(attr0_desc in test_config.__doc__)
    self.assertTrue(
        test_config.__doc__.count(".. py:attribute:: attr0\n") == 1)

    # Basic config.
    nested1_desc = '\n'.join([
        "      .. _nested1:", "      .. py:attribute:: nested1", "      ",
        "         This is the docstring for the nested1 category"
    ])
    self.assertTrue(nested1_desc in test_config.__doc__)
    self.assertTrue(
        test_config.__doc__.count(".. py:attribute:: nested1\n") == 1)

    # Deprecated config.
    nested2_desc = '\n'.join([
        "         .. _nested1.nested2:",
        "         .. py:attribute:: nested1.nested2", "         ",
        "            .. note::",
        "               DEPRECATED: Category is no longer relevant.",
        "            This is a docstring that tests deprecating a nested config"
    ])
    self.assertTrue(nested2_desc in test_config.__doc__)
    self.assertTrue(
        test_config.__doc__.count(".. py:attribute:: nested1.nested2\n") == 1)

    # Deprecated attribute.
    attr1_desc = '\n'.join([
        "         .. _nested1.attr1:",
        "         .. py:attribute:: nested1.attr1",
        "            :type: typing.Union[int, list]", "            :value: 1",
        "         ", "            .. note::",
        "               DEPRECATED: Attribute has been removed as it doesn't"
        " have an effect.", "            This is the docstring for attr1"
    ])
    self.assertTrue(attr1_desc in test_config.__doc__)
    self.assertTrue(
        test_config.__doc__.count(".. py:attribute:: nested1.attr1\n") == 1)

    # Indirectly deprecated attribute (through parent config deprecation).
    attr2_desc = '\n'.join([
        "            .. _nested1.nested2.attr2:",
        "            .. py:attribute:: nested1.nested1.nested2.attr2",
        "               :type: int", "               :value: 1",
        "            ", "               .. note::",
        "                  DEPRECATED: Category is no longer relevant.",
        "               No description provided."
    ])
    self.assertTrue(attr2_desc in test_config.__doc__)
    self.assertTrue(
        test_config.__doc__.count(
            ".. py:attribute:: nested1.nested1.nested2.attr2\n") == 1)

    # pylint: enable=unsupported-membership-test

    # Check doc generation is idempotent on the initialization
    test_config = TestConfig()
    self.assertEqual(test_config.__doc__.count(attr0_desc), 1)
    self.assertEqual(test_config.__doc__.count(nested1_desc), 1)
    self.assertEqual(test_config.__doc__.count(nested2_desc), 1)
    self.assertEqual(test_config.__doc__.count(attr1_desc), 1)
    self.assertEqual(test_config.__doc__.count(attr2_desc), 1)

  def testToProtobuf(self):
    """ Check that to_protobuf is recursively called on the entire config """
    # In the test case, the base class implements _to_protobuf.
    # Some of the nested configs implement it, but others don't. The ones that
    # don't, however, contain nested configs that do.
    # We need to make sure that all configs have their _to_protobuf called in
    # this instance.

    # Use a dictionary to mock the protobuf object.
    attrs = {}
    test_config = TestConfig()
    test_config._to_protobuf(attrs)

    self.assertTrue('base_config' in attrs)
    self.assertEqual(attrs['base_config'], [1])
    self.assertTrue('nested1' not in attrs)
    self.assertTrue('nested2' in attrs)
    self.assertEqual(attrs['nested2'], [1])
    self.assertTrue('nested3' in attrs)
    self.assertEqual(attrs['nested3'], ["test", 1])

  def testParallelConfigs(self):
    """ Check two configs used in parallel don't affect eachother """
    test_config1 = TestConfig()
    test_config2 = TestConfig()

    # Change an attribute on one and the other shouldn't change
    self.assertEqual(test_config1.attr0, 1)
    self.assertEqual(test_config2.attr0, 1)
    test_config1.attr0 = 2
    self.assertEqual(test_config1.attr0, 2)
    self.assertEqual(test_config2.attr0, 1)

  def testParallelConfigsDeprecation(self):
    """ Check two configs don't interfere with each other's deprecation """
    @ipu.config.deprecate_config_attribute("attr0", "config1 message")
    class Config1(ipu.config._ConfigBase):
      def __init__(self):
        self.attr0: int = 1
        self._finalize_base_config()

    @ipu.config.deprecate_config_attribute("attr0", "config2 message")
    class Config2(ipu.config._ConfigBase):
      def __init__(self):
        self.attr0: int = 1
        self._finalize_base_config()

    config1 = Config1()
    config2 = Config2()
    self.assertEqual(
        config1.get_attribute_metadata("attr0").deprecated_msg,
        "config1 message")
    self.assertEqual(
        config2.get_attribute_metadata("attr0").deprecated_msg,
        "config2 message")

  def testDeepCopy(self):
    """ Check we can deepcopy a config """
    test_config = TestConfig()
    test_config.attr0 = [5]
    test_config.nested1.attr1 = [2]
    config_copy = copy.deepcopy(test_config)
    self.assertFalse(test_config.attr0 is config_copy.attr0)
    self.assertFalse(test_config.nested1.attr1 is config_copy.nested1.attr1)

  @parameterized.parameters([True, False])
  def testToDictAndToJson(self, is_json):
    test_config = TestConfig()
    test_config.attr0 = 1234
    test_config.nested1.attr1 = 5678

    if is_json:
      json_cfg = test_config.to_json()
      attrs = json.loads(json_cfg)
    else:
      attrs = test_config.to_dict()

    self.assertTrue("attr0" in attrs)
    self.assertEqual(attrs["attr0"], 1234)
    self.assertTrue("nested1.attr1" in attrs)
    self.assertEqual(attrs["nested1.attr1"], 5678)

  @parameterized.parameters([True, False])
  def testFromDictAndFromJson(self, is_json):
    attrs = {"attr0": 12, "nested1.attr1": 34, "nested1.nested2.attr2": 56}

    test_config = TestConfig()
    test_config.attr0 = 5
    self.assertEqual(test_config.attr0, 5)
    test_config.nested1.attr1 = 5
    self.assertEqual(test_config.nested1.attr1, 5)
    test_config.nested1.nested2.attr2 = 5
    self.assertEqual(test_config.nested1.nested2.attr2, 5)

    if is_json:
      json_cfg = json.dumps(attrs)
      test_config.from_json(json_cfg)
    else:
      test_config.from_dict(attrs)

    self.assertEqual(test_config.attr0, 12)
    self.assertEqual(test_config.nested1.attr1, 34)
    self.assertEqual(test_config.nested1.nested2.attr2, 56)

  @parameterized.parameters([True, False])
  def testToDictAndToJsonOnEnums(self, is_json):
    test_config = TestConfig()
    test_config.attr0 = 12
    test_config.nested1.attr1 = 3
    test_config.nested1.nested2.attr2 = 5
    test_config.nested1.nested2.nested4.attr7 = TestEnum.BANANA

    if is_json:
      json_cfg = test_config.to_json()
      actual = json.loads(json_cfg)
    else:
      actual = test_config.to_dict()

    self.assertEqual(actual["attr0"], 12)
    self.assertIsInstance(actual["nested1.attr1"], int)
    self.assertIs(actual["nested1.attr1"], 3)
    self.assertEqual(actual["nested1.nested2.attr2"], 5)
    self.assertEqual(actual["nested1.nested2.nested4.attr7"], 3)

  @parameterized.parameters([True, False])
  def testFromDictAndFromJsonOnEnums(self, is_json):
    attrs = {
        "attr0": 12,
        "nested1.attr1": 2,
        "nested1.nested2.attr2": 56,
        "nested1.nested2.nested4.attr7": 2
    }

    test_config = TestConfig()
    test_config.attr0 = 5
    test_config.nested1.attr1 = 5
    test_config.nested1.nested2.attr2 = 5

    if is_json:
      json_cfg = json.dumps(attrs)
      test_config.from_json(json_cfg)
    else:
      test_config.from_dict(attrs)

    self.assertEqual(test_config.attr0, 12)
    self.assertIs(test_config.nested1.attr1, 2)
    self.assertEqual(test_config.nested1.nested2.attr2, 56)
    self.assertIsInstance(test_config.nested1.nested2.nested4.attr7, TestEnum)
    self.assertEqual(test_config.nested1.nested2.nested4.attr7,
                     TestEnum.ORANGE)

  def testRepr(self):
    expected = [
        "  attr0 = 12", "  attr6 = ([1.0],)", "  nested1.attr1 = 34",
        "  nested1.nested2.attr2 = 56",
        "  nested1.nested2.nested4.attr4 = False",
        "  nested1.nested2.nested4.attr7 = TestEnum.PEAR",
        "  nested1.nested2.attr5 = 1", "  nested1.nested3.attr3 = 'test'",
        "  nested1.nested3.attr0 = 1"
    ]

    test_config = TestConfig()
    test_config.attr0 = 12
    test_config.nested1.attr1 = 34
    test_config.nested1.nested2.attr2 = 56
    test_config.nested1.nested2.nested4.attr7 = TestEnum.PEAR

    output = repr(test_config)
    output_lines = output.split(os.linesep)

    self.assertEqual(output_lines[0], "TestConfig:")
    for line in expected:
      self.assertIn(line, output_lines[1:])

  def testUnionOfIntAndEnumFails(self):
    with self.assertRaisesRegex(ValueError,
                                "cannot have a Union of int and Enum"):

      class BadConfig(ipu.config._ConfigBase):
        def __init__(self):
          self.bad_config_var: typing.Union[int, enum.Enum] = TestEnum.APPLE

      _ = BadConfig()


# pylint: enable=pointless-string-statement


class IPUConfigTest(test_util.TensorFlowTestCase):
  """
  Tests the ipu.config.IPUConfig class specifically.
  """
  def testCreateConfig(self):
    cfg = ipu.config.IPUConfig()
    self.assertTrue(isinstance(cfg, ipu.utils.IPUConfig))

  def testAutoSelectInteger(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.auto_select_ipus = 2
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 1)
    self.assertEqual(pb.device_config[0].auto_count, 2)

  def testAutoSelectList(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.auto_select_ipus = [4, 4]
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 2)
    self.assertEqual(pb.device_config[0].auto_count, 4)
    self.assertEqual(pb.device_config[1].auto_count, 4)

  def testAutoSelectListManyDevices(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.auto_select_ipus = [2, 3, 4, 5]
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 4)
    self.assertEqual(pb.device_config[0].auto_count, 2)
    self.assertEqual(pb.device_config[1].auto_count, 3)
    self.assertEqual(pb.device_config[2].auto_count, 4)
    self.assertEqual(pb.device_config[3].auto_count, 5)

  def testAutoSelectTuple(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.auto_select_ipus = tuple([2, 2])
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 2)
    self.assertEqual(pb.device_config[0].auto_count, 2)
    self.assertEqual(pb.device_config[1].auto_count, 2)

  def testAutoSelectBoolFails(self):
    cfg = ipu.config.IPUConfig()
    with self.assertRaisesRegex(
        TypeError,
        "Trying to set auto_select_ipus to True, but it must be of type"):
      cfg.auto_select_ipus = True

  def testAutoSelectAndSelectIPUsMutuallyExclusive(self):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = [1, 1]
    cfg.select_ipus = [1, 2]
    with self.assertRaisesRegex(
        Exception,
        "Only one of `auto_select_ipus` and `select_ipus` can be set."):
      cfg._create_protobuf()

  @tu.skip_on_hw
  def testAutoSelectOnlyWhenUsingIPUModel(self):
    cfg = ipu.config.IPUConfig()
    cfg.select_ipus = [1, 2]
    with self.assertRaisesRegex(
        Exception, "When using the IPUModel, the devices to attach to"):
      cfg.auto_select_ipus = []
      cfg._create_protobuf()

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testSelectIPUsInteger(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.select_ipus = 1
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 1)
    self.assertEqual(pb.device_config[0].cfg_index, 1)

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testSelectIPUsList(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.select_ipus = [1, 2]
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 2)
    self.assertEqual(pb.device_config[0].cfg_index, 1)
    self.assertEqual(pb.device_config[1].cfg_index, 2)

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testSelectIPUsTuple(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 0)
    cfg.select_ipus = tuple([1, 2])
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.device_config), 2)
    self.assertEqual(pb.device_config[0].cfg_index, 1)
    self.assertEqual(pb.device_config[1].cfg_index, 2)

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testSelectIPUsBoolFails(self):
    cfg = ipu.config.IPUConfig()
    with self.assertRaisesRegex(
        TypeError,
        "Trying to set select_ipus to True, but it must be of type"):
      cfg.select_ipus = True

  def testAllowRecompute(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertFalse(pb.speed_size_config.allow_recompute)
    cfg.allow_recompute = True
    pb = cfg._create_protobuf()
    self.assertTrue(pb.speed_size_config.allow_recompute)

  def testSelectionOrder(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.selection_order, IpuSelectionOrder.AUTO)
    cfg.selection_order = ipu.utils.SelectionOrder.SNAKE
    pb = cfg._create_protobuf()
    self.assertEqual(pb.selection_order, IpuSelectionOrder.SNAKE)

  def testSerializationOutputFolder(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.serialization_folder, "")
    cfg.serialization_output_folder = "/a/test/path"
    pb = cfg._create_protobuf()
    self.assertEqual(pb.serialization_folder, "/a/test/path")

  def testCompilationPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.compilation_options), 0)
    cfg.compilation_poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.compilation_options[0].option, 'A')
    self.assertEqual(pb.compilation_options[0].value, 'B')
    self.assertEqual(pb.compilation_options[1].option, 'C')
    self.assertEqual(pb.compilation_options[1].value, 'D')

  def testGCLPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.gcl_options), 0)
    cfg.gcl_poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.gcl_options[0].option, 'A')
    self.assertEqual(pb.gcl_options[0].value, 'B')
    self.assertEqual(pb.gcl_options[1].option, 'C')
    self.assertEqual(pb.gcl_options[1].value, 'D')

  def testConvolutionsPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.convolution_options), 0)
    cfg.convolutions.poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.convolution_options[0].option, 'A')
    self.assertEqual(pb.convolution_options[0].value, 'B')
    self.assertEqual(pb.convolution_options[1].option, 'C')
    self.assertEqual(pb.convolution_options[1].value, 'D')

  def testSlicesPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.slice_options), 0)
    cfg.slices.poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.slice_options[0].option, 'A')
    self.assertEqual(pb.slice_options[0].value, 'B')
    self.assertEqual(pb.slice_options[1].option, 'C')
    self.assertEqual(pb.slice_options[1].value, 'D')

  def testDeviceConnectionVersion(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_version, "")
    cfg.device_connection.version = "ipu2"
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_version, "ipu2")

  def testDeviceConnectionType(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.device_connection_type, IpuDeviceConnectionType.ALWAYS)
    cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
    pb = cfg._create_protobuf()
    self.assertEqual(pb.device_connection_type,
                     IpuDeviceConnectionType.ON_DEMAND)

  def testDeviceConnectionTypeVersionRequired(self):
    # Test setting type to PRE_COMPILE or NEVER without setting version
    cfg = ipu.config.IPUConfig()
    cfg.device_connection.version = ""

    cfg.device_connection.type = ipu.utils.DeviceConnectionType.PRE_COMPILE
    with self.assertRaisesRegex(ValueError,
                                "device_connection.version must be set when"):
      cfg._create_protobuf()

    cfg.device_connection.type = ipu.utils.DeviceConnectionType.NEVER
    with self.assertRaisesRegex(ValueError,
                                "device_connection.version must be set when"):
      cfg._create_protobuf()

  def testDeviceConnectionEnableRemoteBuffers(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_remote_buffers_without_device, False)
    cfg.device_connection.enable_remote_buffers = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_remote_buffers_without_device, True)

  def testExperimentalAlwaysRearrangeCopiesOnTheHost(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.always_rearrange_copies_on_the_host,
                     False)
    cfg.experimental.always_rearrange_copies_on_the_host = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.always_rearrange_copies_on_the_host,
                     True)

  def testExperimentalEnableRemoteBufferEmbedding(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_experimental_remote_buffer_embedding, False)
    cfg.experimental.enable_remote_buffer_embedding = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_experimental_remote_buffer_embedding, True)

  def testExperimentalEnablePrngStability(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_experimental_prng_stability, False)
    cfg.experimental.enable_prng_stability = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_experimental_prng_stability, True)

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testExperimentalMultiReplicaDistributionProcessCount(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.multi_replica_process_count, 0)
    cfg.experimental.multi_replica_distribution.process_count = 2
    pb = cfg._create_protobuf()
    self.assertEqual(pb.multi_replica_process_count, 2)

  @tu.skip_on_hw
  def testExperimentalMultiReplicaDistributionProcessCountAndIPUModel(self):
    # Check multi_replica_distribution.process_count can't be set when using the
    # IPU model
    cfg = ipu.config.IPUConfig()
    cfg.experimental.multi_replica_distribution.process_count = 2
    with self.assertRaisesRegex(
        Exception,
        "Multi-replica distribution is not supported on the IPU model."):
      cfg._create_protobuf()

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testExperimentalMultiReplicaDistributionProcessIndex(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.multi_replica_process_index, 0)
    cfg.experimental.multi_replica_distribution.process_index = 1
    pb = cfg._create_protobuf()
    self.assertEqual(pb.multi_replica_process_index, 1)

  @tu.test_uses_ipus(1, allow_ipu_model=False)
  def testExperimentalMultiReplicaDistributionProcessIndexOutOfBounds(self):
    cfg = ipu.config.IPUConfig()
    cfg.experimental.multi_replica_distribution.process_count = 1
    cfg.experimental.multi_replica_distribution.process_index = 5
    with self.assertRaisesRegex(
        ValueError,
        "experimental.multi_replica_distribution.process_index must be in"
        " the range"):
      cfg._create_protobuf()
    cfg.experimental.multi_replica_distribution.process_index = -1
    with self.assertRaisesRegex(
        ValueError,
        "experimental.multi_replica_distribution.process_index must be in"
        " the range"):
      cfg._create_protobuf()

  def check_fp_opts(self, pb, values):
    FPOPTS = ['inv', 'div0', 'oflo', 'esr', 'nanoo']
    for fpopt, value in zip(FPOPTS, values):
      self.assertEqual(getattr(pb.floating_point_behaviour, fpopt), value)

  def testFloatingPointBehaviour(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()

    self.check_fp_opts(pb, [False, False, False, False, False])

    cfg.floating_point_behaviour.inv = True
    pb = cfg._create_protobuf()
    self.check_fp_opts(pb, [True, False, False, False, False])

    cfg.floating_point_behaviour.inv = False
    cfg.floating_point_behaviour.oflo = True
    cfg.floating_point_behaviour.nanoo = True
    pb = cfg._create_protobuf()
    self.check_fp_opts(pb, [False, False, True, False, True])

  def testFloatingPointBehaviourSetAll(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.check_fp_opts(pb, [False, False, False, False, False])

    cfg.floating_point_behaviour.set_all = True
    pb = cfg._create_protobuf()
    self.check_fp_opts(pb, [True, True, True, True, True])

  def testIOTilesNumIOTiles(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.num_io_tiles, 0)
    cfg.io_tiles.num_io_tiles = 10
    pb = cfg._create_protobuf()
    self.assertEqual(pb.num_io_tiles, 10)

  def testIOTilesPlaceOpsOnIOTiles(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.place_ops_on_io_tiles, False)
    cfg.io_tiles.num_io_tiles = 10
    cfg.io_tiles.place_ops_on_io_tiles = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.place_ops_on_io_tiles, True)

  def testIOTilesPlaceOpsOnIOTilesWhenNoIOTiles(self):
    # Test place_ops_on_io_tiles when there aren't any
    cfg = ipu.config.IPUConfig()
    cfg.io_tiles.num_io_tiles = 0
    cfg.io_tiles.place_ops_on_io_tiles = True
    with self.assertRaisesRegex(ValueError,
                                "Cannot place ops on I/O tiles when"):
      cfg._create_protobuf()

  def testIOTilesAvailableMemoryProportion(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.io_tile_available_memory_proportion, 0.9)
    cfg.io_tiles.available_memory_proportion = 0.1
    pb = cfg._create_protobuf()
    self.assertEqual(pb.io_tile_available_memory_proportion, 0.1)

  def testIPUModelCompileIPUCode(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.compile_ipu_code, True)
    cfg.ipu_model.compile_ipu_code = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.compile_ipu_code, False)

  def testIPUModelTilesPerIPU(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.tiles_per_ipu, 0)
    cfg.ipu_model.tiles_per_ipu = 100
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.tiles_per_ipu, 100)

  def testIPUModelVersion(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.ipu_model_version, "ipu2")
    cfg.ipu_model.version = "ipu1"
    pb = cfg._create_protobuf()
    self.assertEqual(pb.ipu_model_config.ipu_model_version, "ipu1")

  @tu.skip_on_hw
  def testIPUModelVersionRequiredWhenRunningOnIPUModel(self):
    # Test ipu_model.version not set when running on the IPU model
    cfg = ipu.config.IPUConfig()
    with self.assertRaisesRegex(ValueError, "ipu_model.version must be set"):
      cfg.ipu_model.version = ""
      cfg._create_protobuf()
    cfg.ipu_model.version = "ipu1"

  def testMatmulsClearPassType(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.clear_matmul_pass_type, False)
    cfg.matmuls.clear_pass_type = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.clear_matmul_pass_type, True)

  def testMatmulsPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.matmul_options), 0)
    cfg.matmuls.poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.matmul_options[0].option, 'A')
    self.assertEqual(pb.matmul_options[0].value, 'B')
    self.assertEqual(pb.matmul_options[1].option, 'C')
    self.assertEqual(pb.matmul_options[1].value, 'D')

  def testNormsUseStableStatistics(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.use_stable_norm_statistics, False)
    cfg.norms.use_stable_statistics = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.use_stable_norm_statistics, True)

  def testNormsExperimentalDistributedBatchNormReplicaGroupSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.experimental_distributed_batch_norm_replica_group_size,
                     1)
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = 8
    pb = cfg._create_protobuf()
    self.assertEqual(pb.experimental_distributed_batch_norm_replica_group_size,
                     8)

  def testNormsExperimentalDistributedBatchNormReplicaGroupSizeAndIOTiles(
      self):
    # Test norms.experimental.distributed_batch_norm_replica_group_size isn't
    # supported with I/O tiles.
    cfg = ipu.config.IPUConfig()
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = 2
    cfg.io_tiles.num_io_tiles = 10
    with self.assertRaisesRegex(ValueError,
                                "is not supported with I/O tiles."):
      cfg._create_protobuf()

  def testOptimizationsPrefetchDataStreams(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.prefetch_data_streams, True)
    cfg.optimizations.prefetch_data_streams = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.prefetch_data_streams, False)

  def testOptimizationsCombineEmbeddingLookups(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_multi_slice_combiner, False)
    cfg.optimizations.combine_embedding_lookups = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_multi_slice_combiner, True)

  def testOptimizationsCombineMatmuls(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_matmul_combiner, False)
    cfg.optimizations.combine_matmuls = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_matmul_combiner, True)

  def testOptimizationsEnableGraphOutlining(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.disable_graph_outlining, False)
    cfg.optimizations.enable_graph_outlining = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.disable_graph_outlining, True)

  def testOptimizationsMergeInfeedIOCopies(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.merge_infeed_io_copies, True)
    cfg.optimizations.merge_infeed_io_copies = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.merge_infeed_io_copies, False)

  def testOptimizationsMaximumCrossReplicaSumBufferSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_cross_replica_sum_buffer_size, 0)
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_cross_replica_sum_buffer_size, 1024768)

  def testOptimizationsMaximumReduceScatterBufferSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_reduce_scatter_buffer_size, 0)
    cfg.optimizations.maximum_reduce_scatter_buffer_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_reduce_scatter_buffer_size, 1024768)

  def testOptimizationsMaximumInterIPUCopiesBufferSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_inter_ipu_copies_buffer_size, 0)
    cfg.optimizations.maximum_inter_ipu_copies_buffer_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_inter_ipu_copies_buffer_size, 1024768)

  def testOptimizationsMaximumReduceManyBufferSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_reduce_many_buffer_size, 0)
    cfg.optimizations.maximum_reduce_many_buffer_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_reduce_many_buffer_size, 1024768)

  def testOptimizationsMaximumGatherBufferSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_all_gather_buffer_size, 0)
    cfg.optimizations.maximum_all_gather_buffer_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_all_gather_buffer_size, 1024768)

  def testOptimizationsMaximumSendRecvClusterSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_send_recv_cluster_size, 0)
    cfg.optimizations.maximum_send_recv_cluster_size = 1024768
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_send_recv_cluster_size, 1024768)

  def testOptimizationsMinimumRemoteTensorSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.minimum_remote_tensor_size, 128)
    cfg.optimizations.minimum_remote_tensor_size = 64
    pb = cfg._create_protobuf()
    self.assertEqual(pb.minimum_remote_tensor_size, 64)

  def testOptimizationsMergeRemoteBuffers(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.remote_buffer_merging_mode,
                     threestate_pb2.THREESTATE_UNDEFINED)
    cfg.optimizations.merge_remote_buffers = \
        ipu.utils.MergeRemoteBuffersBehaviour.NO_MERGING
    pb = cfg._create_protobuf()
    self.assertEqual(pb.remote_buffer_merging_mode,
                     threestate_pb2.THREESTATE_OFF)

  def testOptimizationsEnableGatherSimplifier(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.disable_gather_simplifier, False)
    cfg.optimizations.enable_gather_simplifier = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.disable_gather_simplifier, True)

  def testOptimizationsTriangularSolveExpanderBlockSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.triangular_solve_expander_block_size, 0)
    cfg.optimizations.triangular_solve_expander_block_size = 100
    pb = cfg._create_protobuf()
    self.assertEqual(pb.triangular_solve_expander_block_size, 100)

  def testOptimizationsCholeskyBlockSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.cholesky_block_size, 0)
    cfg.optimizations.cholesky_block_size = 100
    pb = cfg._create_protobuf()
    self.assertEqual(pb.cholesky_block_size, 100)

  def testOptimizationsEnableFastMath(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.algebraic_simplifier_config.enable_fast_math, False)
    self.assertEqual(pb.enable_fast_math, False)

    cfg.optimizations.math.fast = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.algebraic_simplifier_config.enable_fast_math, True)
    self.assertEqual(pb.enable_fast_math, False)

    cfg.optimizations.enable_fast_math = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.algebraic_simplifier_config.enable_fast_math, True)
    self.assertEqual(pb.enable_fast_math, True)

    cfg.optimizations.math.fast = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.algebraic_simplifier_config.enable_fast_math, False)
    self.assertEqual(pb.enable_fast_math, True)

    cfg.optimizations.enable_fast_math = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.algebraic_simplifier_config.enable_fast_math, False)
    self.assertEqual(pb.enable_fast_math, False)

  def testOptimizationsEnableDynamicSliceReplacement(self):
    cfg = ipu.config.IPUConfig()

    cfg.optimizations.enable_dynamic_slice_replacement = False
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_dynamic_slice_replacement, False)

    cfg.optimizations.enable_dynamic_slice_replacement = True
    pb = cfg._create_protobuf()
    self.assertEqual(pb.enable_dynamic_slice_replacement, True)

  def testPoolingPoplarOptions(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(len(pb.pooling_options), 0)
    cfg.pooling.poplar_options = {'A': 'B', 'C': 'D'}
    pb = cfg._create_protobuf()
    self.assertEqual(pb.pooling_options[0].option, 'A')
    self.assertEqual(pb.pooling_options[0].value, 'B')
    self.assertEqual(pb.pooling_options[1].option, 'C')
    self.assertEqual(pb.pooling_options[1].value, 'D')

  def testSchedulingAlgorithm(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.scheduler_selection,
                     IpuSchedulingAlgorithm.CHOOSE_BEST)
    cfg.scheduling.algorithm = ipu.utils.SchedulingAlgorithm.SHORTEST_PATH
    pb = cfg._create_protobuf()
    self.assertEqual(pb.speed_size_config.scheduler_selection,
                     IpuSchedulingAlgorithm.SHORTEST_PATH)

  def testSchedulingMaximumSchedulerLookaheadDepth(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_scheduler_lookahead_depth, 5)
    cfg.scheduling.maximum_scheduler_lookahead_depth = 100
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_scheduler_lookahead_depth, 100)

  def testSchedulingMaximumSchedulerSearchSpaceSize(self):
    cfg = ipu.config.IPUConfig()
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_scheduler_search_space_size, 64)
    cfg.scheduling.maximum_scheduler_search_space_size = 100
    pb = cfg._create_protobuf()
    self.assertEqual(pb.max_scheduler_search_space_size, 100)

  def testCheckMetadata(self):
    """ Check that a couple of the attributes' metadata are correct """
    cfg = ipu.config.IPUConfig()

    md = cfg.get_attribute_metadata("auto_select_ipus")
    self.assertEqual(md.name, "auto_select_ipus")
    self.assertTrue("Configure the IPUs to be used " in md.__doc__)
    self.assertEqual(
        md.type, "typing.Union[int, typing.List[int], typing.Tuple[int, ...]]")
    self.assertEqual(md.default, [])
    self.assertEqual(md.deprecated, False)
    self.assertEqual(md.deprecated_msg, None)
    self.assertEqual(md._depth, 1)

    md = cfg.get_attribute_metadata("ipu_model.version")
    self.assertEqual(md.name, "ipu_model.version")
    self.assertTrue("Specify the IPU version to be used by " in md.__doc__)
    self.assertEqual(md.type, "str")
    self.assertEqual(md.default, "ipu2")
    self.assertEqual(md.deprecated, False)
    self.assertEqual(md.deprecated_msg, None)
    self.assertEqual(md._depth, 2)


# pylint: enable=protected-access

if __name__ == "__main__":
  googletest.main()

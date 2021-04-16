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
import typing

from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


# pylint: disable=pointless-string-statement
class TestConfigNested4(ipu.config.ConfigBase):  # pylint: disable=protected-access
  def __init__(self):
    self.attr4 = False


class TestConfigNested3(ipu.config.ConfigBase):  # pylint: disable=protected-access
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
class TestConfigNested2(ipu.config.ConfigBase):  # pylint: disable=protected-access
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


@ipu.config.deprecate_config_attribute(
    "attr1", "Attribute has been removed as it doesn't have an effect.")
@ipu.config.deprecate_config_attribute("nested2",
                                       "Category is no longer relevant.")
class TestConfigNested1(ipu.config.ConfigBase):  # pylint: disable=protected-access
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


class TestConfig(ipu.config.ConfigBase):  # pylint: disable=protected-access
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


class ConfigBaseTest(test_util.TensorFlowTestCase):
  """
  Tests the basic mechanics of the ipu.config.ConfigBase class using a custom
  config.
  """
  @test_util.deprecated_graph_mode_only
  def testGetFullName(self):
    """
    A _ConfigBase can contain arbitrarily nested _ConfigBase attributes.
    These nested classes should be invisible to the user, as they're an
    implementation detail. The user only knows about the root _ConfigBase, so
    all attribute names should be relative to this root.
    """
    test_config = TestConfig()
    # pylint: disable=protected-access
    self.assertEqual(test_config._get_full_name('attr0'), 'attr0')
    self.assertEqual(test_config.nested1._get_full_name('attr1'),
                     'nested1.attr1')
    self.assertEqual(test_config.nested1.nested2._get_full_name('attr2'),
                     'nested1.nested2.attr2')
    self.assertEqual(test_config.nested1.nested3._get_full_name('attr3'),
                     'nested1.nested3.attr3')
    self.assertEqual(test_config.nested1.nested3._get_full_name('attr0'),
                     'nested1.nested3.attr0')
    # pylint: enable=protected-access

  @test_util.deprecated_graph_mode_only
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
      test_config._private_attribute = []  # pylint: disable=protected-access
    with self.assertRaisesRegex(
        ValueError,
        "'_attr_metadata' is not a valid attribute of this config."):
      test_config._attr_metadata = {}  # pylint: disable=protected-access

    # Check that attributes with the same names aren't the same buffer.
    test_config.attr0 = 5
    test_config.nested1.nested3.attr0 = 5
    self.assertEqual(test_config.attr0, test_config.nested1.nested3.attr0)
    test_config.attr0 = 3
    self.assertNotEqual(test_config.attr0, test_config.nested1.nested3.attr0)

  @test_util.deprecated_graph_mode_only
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

  @test_util.deprecated_graph_mode_only
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

  @test_util.deprecated_graph_mode_only
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
      self.assertEqual(md._depth, expected_depth)  # pylint: disable=protected-access

    # Access base attribute metadata and confirm its contents.
    check_metadata(test_config, "attr0", "attr0",
                   "typing.Union[int, str, list]", 1,
                   "This is the docstring for attr0", 1)

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

  @test_util.deprecated_graph_mode_only
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

  @test_util.deprecated_graph_mode_only
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

    # pylint: disable=unsupported-membership-test

    # Basic attribute.
    attr0_desc = '\n'.join([
        "      .. _attr0:", "      .. py:attribute:: attr0",
        "         :type: typing.Union[int, str, list]", "         :value: 1",
        "      ", "         This is the docstring for attr0"
    ])
    self.assertTrue(attr0_desc in test_config.__doc__)

    # Basic config.
    nested1_desc = '\n'.join([
        "      .. _nested1:", "      .. py:attribute:: nested1", "      ",
        "         This is the docstring for the nested1 category"
    ])
    self.assertTrue(nested1_desc in test_config.__doc__)

    # Deprecated config.
    nested2_desc = '\n'.join([
        "         .. _nested1.nested2:",
        "         .. py:attribute:: nested1.nested2", "         ",
        "            .. note::",
        "               DEPRECATED: Category is no longer relevant.",
        "            This is a docstring that tests deprecating a nested config"
    ])
    self.assertTrue(nested2_desc in test_config.__doc__)

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

    # pylint: enable=unsupported-membership-test

    # Check doc generation is idempotent on the initialization
    test_config = TestConfig()
    self.assertEqual(test_config.__doc__.count(attr0_desc), 1)
    self.assertEqual(test_config.__doc__.count(nested1_desc), 1)
    self.assertEqual(test_config.__doc__.count(nested2_desc), 1)
    self.assertEqual(test_config.__doc__.count(attr1_desc), 1)
    self.assertEqual(test_config.__doc__.count(attr2_desc), 1)

  @test_util.deprecated_graph_mode_only
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
    test_config._to_protobuf(attrs)  # pylint: disable=protected-access

    self.assertTrue('base_config' in attrs)
    self.assertEqual(attrs['base_config'], [1])
    self.assertTrue('nested1' not in attrs)
    self.assertTrue('nested2' in attrs)
    self.assertEqual(attrs['nested2'], [1])
    self.assertTrue('nested3' in attrs)
    self.assertEqual(attrs['nested3'], ["test", 1])

  @test_util.deprecated_graph_mode_only
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

  @test_util.deprecated_graph_mode_only
  def testParallelConfigsDeprecation(self):
    """ Check two configs don't interfere with each other's deprecation """
    @ipu.config.deprecate_config_attribute("attr0", "config1 message")
    class Config1(ipu.config.ConfigBase):
      def __init__(self):
        self.attr0: int = 1
        self._finalize_base_config()

    @ipu.config.deprecate_config_attribute("attr0", "config2 message")
    class Config2(ipu.config.ConfigBase):
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


# pylint: enable=pointless-string-statement
if __name__ == "__main__":
  googletest.main()

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
Dataset extractor
~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_dataset_exporters
from tensorflow.python.data.ops import dataset_ops


def dataset_extractor(dataset,
                      num_elements,
                      filename,
                      feed_name,
                      print_stats=True):
  """Allows the user to extract tensors from a `tf.data.Dataset`.

    Args:
      dataset: An instance of `tf.data.Dataset` to extract elements from.
      num_elements: The number of elements to extract from the dataset.
      filename: Where to save the extracted elements to.
      feed_name: Name of the infeed the dataset is associated with.
      print_stats: Whether to print progress messages to the
        console.

    Note:
      All the tuple elements will be saved in the same binary file.

    Returns:
      The operation that will save the elements of the infeed to file.

    Raises:
      TypeError: if `dataset` is not an instance of `tf.data.Dataset`.
      ValueError: if `num_elements` is less than 1.
    """
  if num_elements < 1:
    return ValueError("Expected `num_elements` to be at least 1.")

  if not isinstance(dataset, dataset_ops.DatasetV2):
    return TypeError("Expected `dataset` argument to be of type "
                     "`tf.data.Dataset`, but got %s "
                     "instead." % (str(dataset)))

  try:
    dataset_variant = dataset._variant_tensor  # pylint: disable=protected-access
  except TypeError:
    dataset_variant = dataset._as_variant_tensor  # pylint: disable=protected-access

  struct = dataset_ops.get_structure(dataset)

  return gen_dataset_exporters.dataset_extractor(dataset_variant, print_stats,
                                                 num_elements, filename,
                                                 feed_name,
                                                 **dataset._flat_structure)  # pylint: disable=protected-access


def get_variable_handles(variables):
  """Allows the user to resolve variables to their variable handle names.

    Args:
      variables: List of variables to resolve to handles.

    Returns:
      List of strings giving the variable handles.
  """
  handles = [v.handle for v in variables]
  return gen_dataset_exporters.resource_to_handle_name(variables=handles)


def export_variables(variables,
                     names,
                     filename,
                     metadata=None,
                     print_stats=True):
  """Allows the user to export to file the content of the variables.

    Args:
      variables: List of variables to export to file.
      names: The names to export the variables under. Should normally be the
             names of the variable handles, in order to match anchor names
             exported with the executable (see get_variable_handles()).
      filename: Where to save the extracted elements to.
      metadata: (optional) Path to a metadata file to validate the variables
        against. If provided then the list of variables must exactly match the
        number, type and shape of the parameters or inputs from the metadata
        file.
      print_stats: Whether to print progress messages to the
        console.

    Note:
      All the variables will be saved in the same binary file.

    Returns:
      The operation that will export the content of the variables to file.

    """
  return gen_dataset_exporters.variables_exporter(variables=variables,
                                                  names=names,
                                                  print_stats=print_stats,
                                                  filename=filename,
                                                  metadata_file=metadata or "")


def import_variables(variables,
                     filenames,
                     is_input=True,
                     strict=True,
                     print_stats=True):
  """Allows the user to import from some data files the content of the variables.

    Args:
      variables: List of variables to import from file.
      filenames: List of binary files containing the variables' data.
      is_input: True if the variables are inputs, False if they're parameters.
      strict: If true then the list of variables must exactly match the
        number, type and shape of the parameters or inputs from the metadata
        stored in the binary files.
      print_stats: Whether to print progress messages to the
        console.

    Returns:
      The operation that will load the content of the variables from file.

    """
  names = [v.name for v in variables]
  shapes = [v.shape for v in variables]
  types = [v.dtype for v in variables]
  new_values = gen_dataset_exporters.variables_importer(print_stats,
                                                        is_input,
                                                        filenames,
                                                        names,
                                                        strict,
                                                        output_types=types,
                                                        output_shapes=shapes)
  return [
      variable.assign(new_values[i]) for i, variable in enumerate(variables)
  ]

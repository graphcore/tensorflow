# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Graphcore utility operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.platform import tf_logging as logging
import errno
import hashlib
import imp
import os
import platform
import sys
import threading  # pylint: disable=unused-import

from tensorflow.core.framework import op_def_pb2
from tensorflow.core.lib.core import error_codes_pb2  # pylint: disable=unused-import
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def precompiled_user_op(inputs, op_name, library_path, gp_path=None, outs=None, name=None):
  """
    Call the poplar function 'op_name' located in the shared library at 'library_path'
    as part of the normal tensorflow execution with the given 'inputs'. The shape and 
    type of the output should be specified by 'outs' if it is None it will default to 
    no output. 'outs' should be a dictionary with two elements like so:

    outs = {
          "output_types": [my_types_as_a_list],
          "output_shapes": [my_shapes_as_a_list],
      }
  """

  if outs is None:
      outs = {
          "output_types": [],
          "output_shapes": [],
      }
  gp_path = gp_path if gp_path else ""
  name = name if name else "UserOp/" + op_name
  return gen_poputil_ops.ipu_user_op(inputs, op_name=op_name, library_path=library_path, gp_path=gp_path, name=name, **outs)



def remap(x, name=None):
  """Clone and map the input linearly across the IPU.

  Args:
    x: The tensor to remap.
    name: Optional op name.

  Returns:
    A `Tensor` which is has been linearly mapped across the IPU.
  """

  logging.warning("remap is a Graphcore internal op")
  return gen_poputil_ops.ipu_remap(x, name=name)

def fifo(x, depth, name=None):
  """Introduces a first-in-first-out queue with a fixed depth.

  Args:
    x: The tensor to enqueue.
    depth: The depth of the queue.
    name: Optional op name.

  Returns:
    A `Tensor` which was dequeued from the fifo. This will be `x` at `t - depth`.
    The first `depth` iterations will have unspecified values.
  """
  if (depth < 1):
    return x

  return gen_poputil_ops.ipu_fifo(x, depth=depth, name=name)

def print_tensor(input, name=None):
  """Print the specified input.

  Args:
    input: The tensor to print.
    name: Optional op name.

  Returns:
    An operator that prints the specified input to the standard error. For the
    tensor to be printed one must either return it as part of their XLA function
    which is consumed by ipu_compiler.compile, or include the returned op in the
    input to session.run, or use the operator as a control dependency for
    executed ops by specifying with tf.control_dependencies([print_op]).

  Examples:
    1. Returning the print operation as part of the XLA function:

    .. code-block:: python

       import tensorflow as tf

       from tensorflow.python.ipu.ops import internal_ops
       from tensorflow.python.ipu import scopes

       def my_net(v):
         print_op = internal_ops.print_tensor(v)
         v = v + 1
         return v, print_op

       with scope.ipu_scope("/device:IPU:0"):
         res = ipu_compiler.compile(my_net, inputs=[v])

       ...
       ...

     2. Including the print operation in session.run:

     .. code-block:: python

       import numpy as np
       import tensorflow as tf

       from tensorflow.python.ipu.ops import internal_ops
       from tensorflow.python.ipu import scopes

       with ops.ipu_scope("/device:IPU:0"):
         pa = tf.placeholder(np.float32, [2, 2], name="a")
         print_op = internal.print_tensor(pa)
         x = pa + 1

       with tf.Session() as session:
        result = session.run([x, print_op], feed_dict={pa : np.ones([2, 2])})

       ...
       ...

     3. Using control dependencies:

     .. code-block:: python

       import numpy as np
       import tensorflow as tf

       from tensorflow.python.ipu.ops import internal_ops
       from tensorflow.python.ipu import scopes

       with ops.ipu_scope("/device:IPU:0"):
         pa = tf.placeholder(np.float32, [2, 2], name="a")
         print_op = internal.print_tensor(pa)
         with tf.control_dependencies([print_op]):
           x = pa + 1

       with tf.Session() as session:
        result = session.run(x, feed_dict={pa : np.ones([2, 2])})

       ...
       ...

  """

  return gen_poputil_ops.ipu_print_tensor(input, name=name)

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import time

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent


def enable_ipu_events(ipu_config):  # pylint: disable=missing-param-doc,missing-type-doc
  """
  INTERNAL ONLY.

  Internal wrapper which enables IPU events on an IPUConfig by modifying the
  IPUConfig's _create_protobuf function to turn them on.
  """
  orig_create_protobuf = ipu_config._create_protobuf  # pylint: disable=protected-access

  def _create_protobuf_wrapper():
    pb = orig_create_protobuf()
    pb.profiling.enable_ipu_trace_events = True
    pb.profiling.enable_compilation_trace = True
    pb.profiling.enable_io_trace = True
    return pb

  # Use the default __setattr__ as IPUConfig's __setattr__ is overridden.
  object.__setattr__(ipu_config, '_create_protobuf', _create_protobuf_wrapper)


def extract_all_strings_from_event_trace(events):  # pylint: disable=missing-type-doc,missing-return-type-doc
  """Extract a concatenation of all data strings from an IPU event trace.

  Args:
    events: An array of IPU events.

  Returns:
    A string containing the concatenation of all of the data fields of the
    events.

  """
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)

    result = result + ("-" * 70) + "\n=> @ " + \
             time.strftime('%F %T %z', time.localtime(evt.timestamp)) + ": "

    if evt.type == IpuTraceEvent.COMPILE_BEGIN:
      evt_str = "Compile begin: " + \
                evt.compile_begin.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.COMPILE_END:
      evt_str = "Compile end: " + \
                evt.compile_end.module_name.decode('utf-8') + "\n" + \
                "Duration: " + str(evt.compile_end.duration)
    elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
      evt_str = "Host->Device\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
      evt_str = "Device->Host\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.LOAD_ENGINE:
      evt_str = "Load engine: " + \
                evt.load_engine.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.EXECUTE:
      evt_str = "Execute: " + \
                evt.execute.module_name.decode('utf-8')
    else:
      evt_str = "Unknown event"

    result = result + evt_str + '\n'

  return result


def extract_all_types_from_event_trace(events):  # pylint: disable=missing-type-doc,missing-return-type-doc
  """Return a list of the types of each event in an event trace tensor

  Args:
    events: A tensor containing a list of IPU events as protobuf strings

  Returns:
    A list containing the type of each event
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt.type]
  return result


def extract_all_events(events):  # pylint: disable=missing-type-doc,missing-return-type-doc
  """Extract a list containing each event as an event object

  Args:
    events: A tensor containing a list of IPU events as protobuf strings

  Returns:
    A list containing IpuTraceEvent objects
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt]
  return result

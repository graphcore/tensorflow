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

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "test_utils",
    srcs = [
        "__init__.py",
        "dataset_utils.py",
        "events_utils.py",
        "hardware_test_utils.py",
        "report_helper.py",
        "session_utils.py",
        "test_case_extensions.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/plugin/poplar:backend_protos_py",
        "//tensorflow/compiler/plugin/poplar/ops:ipu_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ipu:config_lib",
    ],
)

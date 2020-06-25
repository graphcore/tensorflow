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
"""Runs a test file as multiple processes using mpirun."""

import subprocess
import sys
import os


def main():
  if len(sys.argv) < 3:
    sys.stderr.write("Error: Missing required arguments\n")
    sys.stderr.write("Usage: {} num_processes test_file [args...]\n".format(
        sys.argv[0]))
    sys.exit(1)

  num_processes = sys.argv[1]
  test_file = sys.argv[2]
  remaining_args = sys.argv[3:]

  # Add the MPI library dirs on the LD_LIBRARY_PATH, as these are not
  # always searched by default (e.g. on CentOS).
  libdirs = subprocess.check_output(["mpic++", "--showme:libdirs"])
  libdirs = libdirs.decode().strip().replace(" ", ":")
  ld_library_path = "{}:{}".format(os.environ["LD_LIBRARY_PATH"], libdirs)

  # The buildbot runs as root, so let's allow that.
  command = [
      "mpirun", "--allow-run-as-root", "--tag-output", "-x",
      "LD_LIBRARY_PATH=" + ld_library_path, "-np", num_processes,
      sys.executable, test_file
  ]

  subprocess.check_call(command + remaining_args)


if __name__ == "__main__":
  main()

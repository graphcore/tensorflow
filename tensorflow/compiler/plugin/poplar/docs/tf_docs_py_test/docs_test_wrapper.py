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
"""Runs a documentation example."""

import subprocess
import sys
import os
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=str)
  parser.add_argument("--num-ipus", type=int, default=0)
  args, remaining_args = parser.parse_known_args()

  num_available_ipus = int(os.getenv("TF_IPU_COUNT", "0"))
  if num_available_ipus < args.num_ipus:
    print(
        f"Skipping: The documentation example requires {args.num_ipus} IPUs, "
        f"but only {num_available_ipus} are available.",
        file=sys.stderr)
    return

  python_interpreter = "python3"
  command = [python_interpreter, args.source] + remaining_args
  subprocess.check_call(command)


if __name__ == "__main__":
  main()

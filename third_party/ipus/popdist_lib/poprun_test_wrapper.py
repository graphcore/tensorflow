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
import argparse
import os
import subprocess
import sys


def main():
  mpirun_path = sys.argv[1]
  poprun_command = sys.argv[2:]

  # Parse the arguments we need for figuring out the number of IPUs required.
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-replicas", type=int, default=1)
  parser.add_argument("--ipus-per-replica", type=int, default=1)
  args, _ = parser.parse_known_args(args=poprun_command)

  num_required_ipus = args.num_replicas * args.ipus_per_replica
  num_available_ipus = int(os.getenv("TF_IPU_COUNT", 0))
  if num_required_ipus > num_available_ipus:
    print(
        f"Skipping: The poprun test requires {num_required_ipus} IPUs "
        f"({args.num_replicas} replicas times {args.ipus_per_replica} "
        f"IPUs per replica), but only {num_available_ipus} are available.",
        file=sys.stderr)
    return

  env = os.environ.copy()
  # Make sure that the desired mpirun binary is found first on the PATH.
  env["PATH"] = "{}:{}".format(os.path.dirname(mpirun_path), env["PATH"])
  subprocess.check_call(poprun_command, env=env)


if __name__ == "__main__":
  main()

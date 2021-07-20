#!/usr/bin/env bash
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
# External `common.sh`

# Keep in sync with tensorflow_estimator and configure.py.
# LINT.IfChange
LATEST_BAZEL_VERSION=3.1.0
# LINT.ThenChange(
#   //tensorflow/opensource_only/configure.py,
#   //tensorflow_estimator/google/kokoro/common.sh,
#   //tensorflow/tools/ci_build/install/install_bazel.sh,
#   //tensorflow/tools/ci_build/install/install_bazel_from_source.sh)

# Run flaky functions with retries.
# run_with_retry cmd
function run_with_retry {
  eval "$1"
  # If the command fails retry again in 60 seconds.
  if [[ $? -ne 0 ]]; then
    sleep 60
    eval "$1"
  fi
}

function die() {
  echo "$@" 1>&2 ; exit 1;
}

# A small utility to run the command and only print logs if the command fails.
# On success, all logs are hidden.
function readable_run {
  # Disable debug mode to avoid printing of variables here.
  set +x
  result=$("$@" 2>&1) || die "$result"
  echo "$@"
  echo "Command completed successfully at $(date)"
  set -x
}

# LINT.IfChange
# Redirect bazel output dir b/73748835
function set_bazel_outdir {
  mkdir -p /tmpfs/bazel_output
  export TEST_TMPDIR=/tmpfs/bazel_output
}

# Downloads bazelisk to ~/bin as `bazel`.
function install_bazelisk {
  date
  case "$(uname -s)" in
    Darwin) local name=bazelisk-darwin-amd64 ;;
    Linux)  local name=bazelisk-linux-amd64  ;;
    *) die "Unknown OS: $(uname -s)" ;;
  esac
  mkdir -p "$HOME/bin"
  wget --no-verbose -O "$HOME/bin/bazel" \
      "https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/$name"
  chmod u+x "$HOME/bin/bazel"
  if [[ ! ":$PATH:" =~ :"$HOME"/bin/?: ]]; then
    PATH="$HOME/bin:$PATH"
  fi
  set_bazel_outdir
  which bazel
  bazel version
  date
}

# Install the given bazel version on linux
function update_bazel_linux {
  if [[ -z "$1" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  else
    BAZEL_VERSION=$1
  fi
  rm -rf ~/bazel
  mkdir ~/bazel

  pushd ~/bazel
  readable_run wget https://github.com/bazelbuild/bazel/releases/download/"${BAZEL_VERSION}"/bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  chmod +x bazel-*.sh
  ./bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh --user
  rm bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  popd

  PATH="/home/kbuilder/bin:$PATH"
  set_bazel_outdir
  which bazel
  bazel version
}
# LINT.ThenChange(
#   //tensorflow_estimator/google/kokoro/common.sh)

function install_ubuntu_16_pip_deps {
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

  # LINT.IfChange(linux_pip_installations)
  # To have reproducible builds, these dependencies should be pinned always.
  # Prefer pinning to the same version as in setup.py
  # First, upgrade pypi wheels
  "${PIP_CMD}" install --user --upgrade setuptools pip wheel
  # Now, install the deps, as listed in setup.py
  "${PIP_CMD}" install --user 'absl-py ~= 0.10'
  "${PIP_CMD}" install --user 'astunparse ~= 1.6.3'
  "${PIP_CMD}" install --user 'flatbuffers ~= 1.12.0'
  "${PIP_CMD}" install --user 'google_pasta ~= 0.2'
  "${PIP_CMD}" install --user 'h5py ~= 2.10.0'
  "${PIP_CMD}" install --user 'keras_preprocessing ~= 1.1.2'
  "${PIP_CMD}" install --user 'numpy ~= 1.19.2'
  "${PIP_CMD}" install --user 'opt_einsum ~= 3.3.0'
  "${PIP_CMD}" install --user 'protobuf >= 3.9.2'
  "${PIP_CMD}" install --user 'six ~= 1.15.0'
  "${PIP_CMD}" install --user 'termcolor ~= 1.1.0'
  "${PIP_CMD}" install --user 'typing_extensions ~= 3.7.4'
  "${PIP_CMD}" install --user 'wheel ~= 0.35'
  "${PIP_CMD}" install --user 'wrapt ~= 1.12.1'
  # We need to pin gast dependency exactly
  "${PIP_CMD}" install --user 'gast == 0.3.3'
  # Finally, install tensorboard and estimator
  # Note that here we want the latest version that matches (b/156523241)
  "${PIP_CMD}" install --user --upgrade 'tb-nightly ~= 2.4.0.a'
  "${PIP_CMD}" install --user --upgrade 'tensorflow_estimator ~= 2.3.0'
  # Test dependencies
  "${PIP_CMD}" install --user 'grpcio ~= 1.32.0'
  "${PIP_CMD}" install --user 'portpicker ~= 1.3.1'
  "${PIP_CMD}" install --user 'scipy ~= 1.5.2'
  # LINT.ThenChange(:mac_pip_installations)
  # Need to be addressed later. Unblocking 2.4 branchcut
  "${PIP_CMD}" install --user 'PyYAML ~= 5.3.1'
}

function install_macos_pip_deps {
  # TODO(mihaimaruseac): Remove need for sudo, then this can be merged with
  # above (probably needs to convert to venv too).
  SUDO_CMD=""
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "sudo" ]]; then
      SUDO_CMD="sudo "
    elif [[ "$1" == "pip3.7" ]]; then
      PIP_CMD="python3.7 -m pip"
      SUDO_CMD="sudo -H "
    elif [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

  # LINT.IfChange(mac_pip_installations)
  # To have reproducible builds, these dependencies should be pinned always.
  # Prefer pinning to the same version as in setup.py
  # First, upgrade pypi wheels
  ${PIP_CMD} install --user --upgrade setuptools pip wheel
  # Now, install the deps, as listed in setup.py
  ${PIP_CMD} install --user 'absl-py ~= 0.10'
  ${PIP_CMD} install --user 'astunparse ~= 1.6.3'
  ${PIP_CMD} install --user 'flatbuffers ~= 1.12.0'
  ${PIP_CMD} install --user 'google_pasta ~= 0.2'
  ${PIP_CMD} install --user 'h5py ~= 2.10.0'
  ${PIP_CMD} install --user 'keras_preprocessing ~= 1.1.2'
  ${PIP_CMD} install --user 'numpy ~= 1.19.2'
  ${PIP_CMD} install --user 'opt_einsum ~= 3.3.0'
  ${PIP_CMD} install --user 'protobuf >= 3.9.2'
  ${PIP_CMD} install --user 'six ~= 1.15.0'
  ${PIP_CMD} install --user 'termcolor ~= 1.1.0'
  ${PIP_CMD} install --user 'typing_extensions ~= 3.7.4'
  ${PIP_CMD} install --user 'wheel ~= 0.35'
  ${PIP_CMD} install --user 'wrapt ~= 1.12.1'
  # We need to pin gast dependency exactly
  ${PIP_CMD} install --user 'gast == 0.3.3'
  # Finally, install tensorboard and estimator
  # Note that here we want the latest version that matches (b/156523241)
  ${PIP_CMD} install --user --upgrade 'tb-nightly ~= 2.4.0.a'
  ${PIP_CMD} install --user --upgrade 'tensorflow_estimator ~= 2.3.0'
  # Test dependencies
  ${PIP_CMD} install --user 'grpcio ~= 1.32.0'
  ${PIP_CMD} install --user 'portpicker ~= 1.3.1'
  ${PIP_CMD} install --user 'scipy ~= 1.5.2'
  # LINT.ThenChange(:linux_pip_installations)
}

function maybe_skip_v1 {
  # If we are building with v2 by default, skip tests with v1only tag.
  if grep -q "build --config=v2" ".bazelrc"; then
    echo ",-v1only"
  else
    echo ""
  fi
}

# Copy and rename a wheel to a new project name.
# Usage: copy_to_new_project_name <whl_path> <new_project_name>, for example
# copy_to_new_project_name test_dir/tf_nightly-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl tf_nightly_cpu
# will create a wheel with the same tags, but new project name under the same
# directory at
# test_dir/tf_nightly_cpu-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl
function copy_to_new_project_name {
  WHL_PATH="$1"
  NEW_PROJECT_NAME="$2"
  PYTHON_CMD="$3"

  # Debugging only, could be removed after we know it works
  echo "copy_to_new_project_name PATH is ${PATH}"

  ORIGINAL_WHL_NAME=$(basename "${WHL_PATH}")
  ORIGINAL_WHL_DIR=$(realpath "$(dirname "${WHL_PATH}")")
  ORIGINAL_PROJECT_NAME="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 1)"
  FULL_TAG="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 2-)"
  NEW_WHL_NAME="${NEW_PROJECT_NAME}-${FULL_TAG}"
  VERSION="$(echo "${FULL_TAG}" | cut -d '-' -f 1)"

  ORIGINAL_WHL_DIR_PREFIX="${ORIGINAL_PROJECT_NAME}-${VERSION}"
  NEW_WHL_DIR_PREFIX="${NEW_PROJECT_NAME}-${VERSION}"

  TMP_DIR="$(mktemp -d)"
  ${PYTHON_CMD} -m wheel unpack "${WHL_PATH}"
  # Debug:
  ls -l
  # End debug
  mv "${ORIGINAL_WHL_DIR_PREFIX}" "${TMP_DIR}"
  # Debug
  ls -l "${TMP_DIR}"
  # End debug
  pushd "${TMP_DIR}/${ORIGINAL_WHL_DIR_PREFIX}"

  mv "${ORIGINAL_WHL_DIR_PREFIX}.dist-info" "${NEW_WHL_DIR_PREFIX}.dist-info"
  if [[ -d "${ORIGINAL_WHL_DIR_PREFIX}.data" ]]; then
    mv "${ORIGINAL_WHL_DIR_PREFIX}.data" "${NEW_WHL_DIR_PREFIX}.data"
  fi

  ORIGINAL_PROJECT_NAME_DASH="${ORIGINAL_PROJECT_NAME//_/-}"
  NEW_PROJECT_NAME_DASH="${NEW_PROJECT_NAME//_/-}"

  # We need to change the name in the METADATA file, but we need to ensure that
  # all other occurences of the name stay the same, otherwise things such as
  # URLs and depedencies might be broken (for example, replacing without care
  # might transform a `tensorflow_estimator` dependency into
  # `tensorflow_gpu_estimator`, which of course does not exist -- except by
  # manual upload of a manually altered `tensorflow_estimator` package)
  sed -i.bak "s/Name: ${ORIGINAL_PROJECT_NAME_DASH}/Name: ${NEW_PROJECT_NAME_DASH}/g" "${NEW_WHL_DIR_PREFIX}.dist-info/METADATA"

  ${PYTHON_CMD} -m wheel pack .
  # Debug
  ls -l
  # End debug
  mv *.whl "${ORIGINAL_WHL_DIR}"
  popd
  rm -rf "${TMP_DIR}"
}

# Create minimalist test XML for web view. It includes the pass/fail status
# of each target, without including errors or stacktraces.
# Remember to "set +e" before calling bazel or we'll only generate the XML for
# passing runs.
function test_xml_summary {
  set +x
  set +e
  mkdir -p "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary"
  # First build the repeated inner XML blocks, since the header block needs to
  # report the number of test cases / failures / errors.
  # TODO(rsopher): handle build breakages
  # TODO(rsopher): extract per-test times as well
  TESTCASE_XML="$(sed -n '/INFO:\ Build\ completed/,/INFO:\ Build\ completed/p' \
    /tmpfs/kokoro_build.log \
    | grep -E '(PASSED|FAILED|TIMEOUT)\ in' \
    | while read -r line; \
      do echo '<testcase name="'"$(echo "${line}" | tr -s ' ' | cut -d ' ' -f 1)"\
          '" status="run" classname="" time="0">'"$( \
        case "$(echo "${line}" | tr -s ' ' | cut -d ' ' -f 2)" in \
          FAILED) echo '<failure message="" type=""/>' ;; \
          TIMEOUT) echo '<failure message="timeout" type=""/>' ;; \
        esac; \
      )"'</testcase>'; done; \
  )"
  NUMBER_OF_TESTS="$(echo "${TESTCASE_XML}" | wc -l)"
  NUMBER_OF_FAILURES="$(echo "${TESTCASE_XML}" | grep -c '<failure')"
  echo '<?xml version="1.0" encoding="UTF-8"?>'\
  '<testsuites name="1"  tests="1" failures="0" errors="0" time="0">'\
  '<testsuite name="Kokoro Summary" tests="'"${NUMBER_OF_TESTS}"\
  '" failures="'"${NUMBER_OF_FAILURES}"'" errors="0" time="0">'\
  "${TESTCASE_XML}"'</testsuite></testsuites>'\
  > "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary/sponge_log.xml"
}

# Create minimalist test XML for web view, then exit.
# Ends script with value of previous command, meant to be called immediately
# after bazel as the last call in the build script.
function test_xml_summary_exit {
  RETVAL=$?
  test_xml_summary
  exit "${RETVAL}"
}

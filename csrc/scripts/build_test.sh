#!/usr/bin/env bash
set -euo pipefail

cmake -S csrc -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_TESTS=ON
cmake --build build --config Debug
ctest --test-dir build --rerun-failed --output-on-failure

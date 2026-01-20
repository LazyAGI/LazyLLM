#!/usr/bin/env bash
set -euo pipefail

cmake -S csrc -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build

#!/usr/bin/env bash
set -euo pipefail

cmake -S csrc -B build-release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

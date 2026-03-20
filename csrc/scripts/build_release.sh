#!/usr/bin/env bash
# Run at LazyLLM/.
set -euo pipefail

cmake -S csrc -B build-release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

# Install into ./lazyllm (prefix=. + LIBRARY DESTINATION lazyllm).
cmake --install build --prefix . --component lazyllm_cpp

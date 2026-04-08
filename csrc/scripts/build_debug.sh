#!/usr/bin/env bash
# Run at LazyLLM/.
set -euo pipefail

cmake -S csrc -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Install into ./lazyllm (local repo copy).
cmake --install build --prefix . --component lazyllm_cpp

# Install into active Python site-packages (editable/venv runtime copy).
PY_PLATLIB="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
cmake --install build --prefix "$PY_PLATLIB" --component lazyllm_cpp

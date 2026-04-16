#!/usr/bin/env bash
# Run at LazyLLM/.
set -euo pipefail

PYTHON="${PYTHON:-python}"

cmake -S csrc -B build \
  -Dpybind11_DIR="$("$PYTHON" -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"

# Install into ./lazyllm (local repo copy).
cmake --install build --prefix . --component lazyllm_cpp

# Install into active Python site-packages (editable/venv runtime copy).
cmake --install build --prefix "$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("platlib") or "")')" --component lazyllm_cpp

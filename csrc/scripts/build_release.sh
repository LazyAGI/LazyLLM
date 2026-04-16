#!/usr/bin/env bash
# Run at LazyLLM/.
set -euo pipefail

cmake -S csrc -B build-release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j"$(nproc)"

# Install into ./lazyllm (local repo copy).
cmake --install build-release --prefix . --component lazyllm_cpp

# Install into active Python site-packages (editable/venv runtime copy).
cmake --install build-release --prefix "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib") or "")')" --component lazyllm_cpp

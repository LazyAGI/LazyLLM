#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void exportDoc(pybind11::module& m);

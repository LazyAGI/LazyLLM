#pragma once

#include <memory>
#include <stdexcept>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

void exportAddDocStr(pybind11::module& m);
void exportDocNode(pybind11::module& m);

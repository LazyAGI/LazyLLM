#pragma once

#include <memory>
#include <stdexcept>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void exportAddDocStr(pybind11::module& m);
void exportDocNode(pybind11::module& m);
void exportNodeTransform(pybind11::module& m);
void exportTextSpliterBase(pybind11::module& m);

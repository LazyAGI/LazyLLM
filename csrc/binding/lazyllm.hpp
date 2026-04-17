#pragma once

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

void exportAddDocStr(pybind11::module& m);
void exportDocNode(pybind11::module& m);
void exportTextSplitterBase(pybind11::module& m);
void exportSentenceSplitter(pybind11::module& m);

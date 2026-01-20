#include "lazyllm.hpp"

#include "docnode.h"

namespace py = pybind11;

PYBIND11_MODULE(lazyllm_cpp, m) {
    m.doc() = "LazyLLM CPP Module.";
    exportDoc(m);

    // prevent document generation
    py::options options;
    options.disable_function_signatures();

    // DocNode
    py::class_<lazyllm::DocNode>(m, "DocNode")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("text"))
        .def("set_text", &lazyllm::DocNode::set_text, py::arg("text"))
        .def("get_text", &lazyllm::DocNode::get_text);
}

#include "lazyllm.hpp"

#include "document_store.hpp"
#include "doc_node.hpp"

#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(lazyllm_cpp, m) {
    m.doc() = "LazyLLM CPP Module.";
    exportAddDocStr(m);

    // prevent document generation
    py::options options;
    options.disable_function_signatures();

    exportDocNode(m);
}

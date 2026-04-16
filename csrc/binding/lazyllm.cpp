#include "lazyllm.hpp"

#include "doc_node.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lazyllm_cpp, m) {
    m.doc() = "LazyLLM CPP Module.";
    exportAddDocStr(m);

    // Prevent document generation
    py::options options;
    options.disable_function_signatures();

    // Export classes
    exportDocNode(m);
    exportTextSplitterBase(m);
    exportSentenceSplitter(m);
}

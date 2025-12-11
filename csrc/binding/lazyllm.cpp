#include "lazyllm.hpp"

PYBIND11_MODULE(lazyllm_cpp, m) {
    m.doc() = "LazyLLM CPP Module.";

    // prevent document generation
    pybind11::options options;
    options.disable_function_signatures();

    exportDoc(m);
}

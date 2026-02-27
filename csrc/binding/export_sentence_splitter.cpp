#include "lazyllm.hpp"

#include "sentence_splitter.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

class PySentenceSplitter final : public lazyllm::SentenceSplitter {
public:
    using lazyllm::SentenceSplitter::SentenceSplitter;

    std::vector<lazyllm::DocNode> transform(const lazyllm::DocNode* node) const override {
        PYBIND11_OVERRIDE(
            std::vector<lazyllm::DocNode>,
            lazyllm::SentenceSplitter,
            transform,
            node
        );
    }
};

} // namespace

void exportSentenceSplitter(py::module& m) {
    py::class_<
        lazyllm::SentenceSplitter,
        lazyllm::TextSplitterBase,
        PySentenceSplitter
    >(m, "SentenceSplitter")
        .def(py::init([](
            std::optional<unsigned> chunk_size,
            std::optional<unsigned> chunk_overlap,
            std::optional<unsigned> num_workers,
            py::object encoding_name
        ) {
            return std::make_unique<lazyllm::SentenceSplitter>(
                chunk_size,
                chunk_overlap,
                num_workers,
                encoding_name.is_none() ? "gpt2" : encoding_name.cast<std::string>());
            }),
            py::arg("chunk_size") = py::none(),
            py::arg("chunk_overlap") = py::none(),
            py::arg("num_workers") = py::none(),
            py::arg("encoding_name") = py::none()
        );
}

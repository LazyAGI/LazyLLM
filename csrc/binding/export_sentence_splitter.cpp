#include "lazyllm.hpp"

#include "sentence_splitter.hpp"

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

class SentenceSplitterCPPImpl : public lazyllm::SentenceSplitter {
public:
    SentenceSplitterCPPImpl(
        unsigned chunk_size,
        unsigned chunk_overlap,
        const std::string& encoding_name = "gpt2")
        : lazyllm::SentenceSplitter(chunk_size, chunk_overlap, encoding_name) {}

    int chunk_size() const { return _chunk_size; }
    void set_chunk_size(int value) { _chunk_size = value; }

    int overlap() const { return _overlap; }
    void set_overlap(int value) { _overlap = value; }


    py::list split_recursive_impl(const std::string& text, int chunk_size) const {
        py::object split_cls = py::module_::import("lazyllm.tools.rag.transform.base").attr("_Split");
        py::list out;
        const auto splits = lazyllm::TextSplitterBase::split_recursive(text, chunk_size);
        for (const auto& split : splits) {
            out.append(split_cls(
                py::arg("text") = std::string(split.view),
                py::arg("is_sentence") = split.is_sentence,
                py::arg("token_size") = split.token_size));
        }
        return out;
    }

    std::vector<std::string> merge_chunks_impl(py::list splits, int chunk_size) const {
        struct OwnedSplit {
            std::string text;
            bool is_sentence;
            int token_size;
        };

        std::vector<OwnedSplit> owned;
        owned.reserve(py::len(splits));
        for (auto item : splits) {
            py::object split = py::reinterpret_borrow<py::object>(item);
            owned.push_back(
                OwnedSplit{
                    split.attr("text").cast<std::string>(),
                    split.attr("is_sentence").cast<bool>(),
                    split.attr("token_size").cast<int>()
                }
            );
        }

        std::vector<lazyllm::ChunkView> views;
        views.reserve(owned.size());
        for (const auto& split : owned) {
            views.push_back(lazyllm::ChunkView{split.text, split.is_sentence, split.token_size});
        }

        return lazyllm::SentenceSplitter::merge_chunks(views, chunk_size);
    }
};

} // namespace

void exportSentenceSplitter(py::module& m) {
    py::class_<SentenceSplitterCPPImpl>(m, "SentenceSplitterCPPImpl", py::dynamic_attr())
        .def(py::init<unsigned, unsigned, const std::string&>(),
            py::arg("chunk_size") = 1024,
            py::arg("chunk_overlap") = 200,
            py::arg("encoding_name") = "gpt2"
        )
        .def_property("_chunk_size", &SentenceSplitterCPPImpl::chunk_size, &SentenceSplitterCPPImpl::set_chunk_size)
        .def_property("_overlap", &SentenceSplitterCPPImpl::overlap, &SentenceSplitterCPPImpl::set_overlap)
        .def("split_text", &SentenceSplitterCPPImpl::split_text, py::arg("text"), py::arg("metadata_size"))
        .def("split_recursive", &SentenceSplitterCPPImpl::split_recursive_impl, py::arg("text"), py::arg("chunk_size"))
        .def("merge_chunks", &SentenceSplitterCPPImpl::merge_chunks_impl, py::arg("splits"), py::arg("chunk_size"));
}

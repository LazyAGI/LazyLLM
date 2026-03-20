#include "lazyllm.hpp"

#include "text_splitter_base.hpp"

#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/gil.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

class TextSplitterBaseCPPImpl : public lazyllm::TextSplitterBase {
public:
    TextSplitterBaseCPPImpl(
        unsigned chunk_size,
        unsigned overlap,
        const std::string& encoding_name = "gpt2")
        : lazyllm::TextSplitterBase(
            static_cast<int>(chunk_size),
            static_cast<int>(overlap),
            encoding_name) {}

    int chunk_size() const { return _chunk_size; }
    void set_chunk_size(int value) { _chunk_size = value; }

    int overlap() const { return _overlap; }
    void set_overlap(int value) { _overlap = value; }


    py::list split_recursive_impl(const std::string& text, int chunk_size) const {
        std::vector<lazyllm::ChunkView> splits;
        {
            py::gil_scoped_release release;
            splits = lazyllm::TextSplitterBase::split_recursive(text, chunk_size);
        }

        py::object split_cls = py::module_::import("lazyllm.tools.rag.transform.base").attr("_Split");
        py::list out;
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

        std::vector<std::string> chunks;
        {
            py::gil_scoped_release release;
            chunks = lazyllm::TextSplitterBase::merge_chunks(views, chunk_size);
        }
        return chunks;
    }
};

} // namespace

void exportTextSpliterBase(py::module& m) {
    py::class_<TextSplitterBaseCPPImpl>(m, "_TextSplitterBaseCPPImpl", py::dynamic_attr())
        .def(py::init<unsigned, unsigned, const std::string&>(),
            py::arg("chunk_size") = 1024,
            py::arg("overlap") = 200,
            py::arg("encoding_name") = "gpt2"
        )
        .def_property("_chunk_size", &TextSplitterBaseCPPImpl::chunk_size, &TextSplitterBaseCPPImpl::set_chunk_size)
        .def_property("_overlap", &TextSplitterBaseCPPImpl::overlap, &TextSplitterBaseCPPImpl::set_overlap)
        .def("split_text", &TextSplitterBaseCPPImpl::split_text, py::arg("text"), py::arg("metadata_size"),
            py::call_guard<py::gil_scoped_release>())
        .def("split_recursive", &TextSplitterBaseCPPImpl::split_recursive_impl, py::arg("text"), py::arg("chunk_size"))
        .def("merge_chunks", &TextSplitterBaseCPPImpl::merge_chunks_impl, py::arg("splits"), py::arg("chunk_size"));
}

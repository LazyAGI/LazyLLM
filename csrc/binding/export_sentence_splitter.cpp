#include "lazyllm.hpp"

#include "sentence_splitter.hpp"

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/gil.h>
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
        std::vector<lazyllm::Chunk> owned;
        owned.reserve(py::len(splits));
        for (auto item : splits) {
            py::object split = py::reinterpret_borrow<py::object>(item);
            owned.push_back(lazyllm::Chunk{
                split.attr("text").cast<std::string>(),
                split.attr("is_sentence").cast<bool>(),
                split.attr("token_size").cast<int>()
            });
        }

        std::vector<std::string> chunks;
        {
            py::gil_scoped_release release;
            chunks = lazyllm::SentenceSplitter::merge_chunks(owned, chunk_size);
        }
        return chunks;
    }
};

} // namespace

void exportSentenceSplitter(py::module& m) {
    auto cls = py::class_<SentenceSplitterCPPImpl>(m, "SentenceSplitterCPPImpl", py::dynamic_attr())
        .def(py::init<unsigned, unsigned, const std::string&>(),
            py::arg("chunk_size") = 1024,
            py::arg("chunk_overlap") = 200,
            py::arg("encoding_name") = "gpt2"
        )
        .def_property("_chunk_size", &SentenceSplitterCPPImpl::chunk_size, &SentenceSplitterCPPImpl::set_chunk_size)
        .def_property("_overlap", &SentenceSplitterCPPImpl::overlap, &SentenceSplitterCPPImpl::set_overlap)
        .def("split_text", &SentenceSplitterCPPImpl::split_text, py::arg("text"), py::arg("metadata_size"),
            py::call_guard<py::gil_scoped_release>())
        .def("_split", &SentenceSplitterCPPImpl::split_recursive_impl, py::arg("text"), py::arg("chunk_size"))
        .def("_merge", &SentenceSplitterCPPImpl::merge_chunks_impl, py::arg("splits"), py::arg("chunk_size"));

    cls.attr("__proxy_methods__") = py::make_tuple("split_text", "_split", "_merge");

    py::dict method_signatures;
    method_signatures["split_text"] = py::make_tuple("text", "metadata_size");
    method_signatures["_split"] = py::make_tuple("text", "chunk_size");
    method_signatures["_merge"] = py::make_tuple("splits", "chunk_size");
    cls.attr("__proxy_method_signatures__") = method_signatures;

    cls.attr("__proxy_attrs__") = py::make_tuple("_chunk_size", "_overlap");

    py::module_ builtins = py::module_::import("builtins");
    py::dict init_param_types;
    init_param_types["chunk_size"] = builtins.attr("int");
    init_param_types["chunk_overlap"] = builtins.attr("int");
    init_param_types["encoding_name"] = builtins.attr("str");
    cls.attr("__init_param_types__") = init_param_types;
}

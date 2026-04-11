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
            chunks = lazyllm::TextSplitterBase::merge_chunks(owned, chunk_size);
        }
        return chunks;
    }

    py::list split_text_impl(const std::string& text, int metadata_size) const {
        std::vector<std::string> chunks;
        {
            py::gil_scoped_release release;
            chunks = lazyllm::TextSplitterBase::split_text(text, metadata_size);
        }

        py::list out;
        for (const auto& chunk : chunks) {
            PyObject* decoded = PyUnicode_DecodeUTF8(
                chunk.data(),
                static_cast<Py_ssize_t>(chunk.size()),
                "replace"
            );
            if (decoded == nullptr) throw py::error_already_set();
            out.append(py::reinterpret_steal<py::str>(decoded));
        }
        return out;
    }
};

} // namespace

void exportTextSplitterBase(py::module& m) {
    auto cls = py::class_<TextSplitterBaseCPPImpl>(m, "_TextSplitterBaseCPPImpl", py::dynamic_attr())
        .def(py::init<unsigned, unsigned, const std::string&>(),
            py::arg("chunk_size") = 1024,
            py::arg("overlap") = 200,
            py::arg("encoding_name") = "gpt2"
        )
        .def_property("_chunk_size", &TextSplitterBaseCPPImpl::chunk_size, &TextSplitterBaseCPPImpl::set_chunk_size)
        .def_property("_overlap", &TextSplitterBaseCPPImpl::overlap, &TextSplitterBaseCPPImpl::set_overlap)
        .def("split_text", &TextSplitterBaseCPPImpl::split_text_impl, py::arg("text"), py::arg("metadata_size"))
        .def("_merge", &TextSplitterBaseCPPImpl::merge_chunks_impl, py::arg("splits"), py::arg("chunk_size"));

    (void)cls;
}

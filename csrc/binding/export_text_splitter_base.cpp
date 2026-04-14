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
        : lazyllm::TextSplitterBase(chunk_size, overlap, encoding_name) {}

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
        .def("split_text", &TextSplitterBaseCPPImpl::split_text_impl, py::arg("text"), py::arg("metadata_size"));

    (void)cls;
}

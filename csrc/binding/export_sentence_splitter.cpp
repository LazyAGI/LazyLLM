#include "lazyllm.hpp"

#include "sentence_splitter.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

class PySentenceSplitter final : public lazyllm::SentenceSplitter {
public:
    using lazyllm::SentenceSplitter::SentenceSplitter;

    std::vector<lazyllm::PDocNode> transform(const lazyllm::PDocNode node) const override {
        PYBIND11_OVERRIDE(
            std::vector<lazyllm::PDocNode>,
            lazyllm::SentenceSplitter,
            transform,
            node
        );
    }
};

py::object GetBaseModule() {
    return py::module_::import("lazyllm.tools.rag.transform.base");
}

py::object GetUnset() {
    return GetBaseModule().attr("_UNSET");
}

bool IsUnset(const py::object& value) {
    return value.is(GetUnset());
}

py::dict GetDefaultParams(const py::object& cls) {
    if (!py::hasattr(cls, "_default_params")) {
        cls.attr("_default_params") = py::dict();
    }
    return cls.attr("_default_params").cast<py::dict>();
}

unsigned ResolveUnsigned(
    const py::object& cls,
    const std::string& param_name,
    const py::object& value,
    unsigned default_value
) {
    if (value.is_none()) return default_value;
    if (!IsUnset(value)) {
        if (py::isinstance<py::int_>(value)) return value.cast<unsigned>();
        throw py::type_error(param_name + " must be an int");
    }

    py::dict defaults = GetDefaultParams(cls);
    if (defaults.contains(param_name.c_str())) {
        py::object v = py::reinterpret_borrow<py::object>(defaults[param_name.c_str()]);
        if (py::isinstance<py::int_>(v)) return v.cast<unsigned>();
    }
    return default_value;
}

std::string Trim(const std::string& input) {
    size_t start = 0;
    size_t end = input.size();
    while (start < end && std::isspace(static_cast<unsigned char>(input[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
    return input.substr(start, end - start);
}

void InitTiktokenTokenizer(
    const py::object& py_self,
    const std::string& encoding_name,
    const py::object& model_name,
    py::object allowed_special,
    py::object disallowed_special
) {
    (void)allowed_special;
    (void)disallowed_special;

    std::string tokenizer_name = encoding_name;
    if (!model_name.is_none()) {
        if (!py::isinstance<py::str>(model_name)) {
            throw py::type_error("model_name must be a string");
        }
        tokenizer_name = model_name.cast<std::string>();
    }
    auto tokenizer = std::make_shared<TiktokenTokenizer>(tokenizer_name);

    py::object encode = py::cpp_function([tokenizer](const std::string& text) {
        return tokenizer->encode(text);
    });
    py::object decode = py::cpp_function([tokenizer](const std::vector<int>& ids) {
        py::bytes raw(tokenizer->decode(ids));
        return raw.attr("decode")("utf-8", "replace");
    });

    py_self.attr("token_encoder") = encode;
    py_self.attr("token_decoder") = decode;
}

} // namespace

void exportSentenceSplitter(py::module& m) {
    py::class_<
        lazyllm::SentenceSplitter,
        lazyllm::TextSplitterBase,
        PySentenceSplitter
    >(m, "SentenceSplitter", py::dynamic_attr())
        .def(py::init([](py::object chunk_size,
                         py::object chunk_overlap,
                         py::object num_workers) {
                py::object cls = py::module_::import("lazyllm.lazyllm_cpp").attr("SentenceSplitter");

                unsigned cs = ResolveUnsigned(cls, "chunk_size", chunk_size, 1024);
                unsigned ov = ResolveUnsigned(cls, "overlap", chunk_overlap, 200);
                unsigned nw = ResolveUnsigned(cls, "num_workers", num_workers, 0);

                if (ov > cs) {
                    throw py::value_error(
                        "Got a larger chunk overlap (" + std::to_string(ov) + ") than chunk size (" +
                        std::to_string(cs) + "), should be smaller.");
                }
                if (cs == 0) {
                    throw py::value_error("chunk size should > 0 and overlap should >= 0");
                }

                return std::make_unique<PySentenceSplitter>(cs, ov, nw, "gpt2");
            }),
            py::arg("chunk_size") = py::none(),
            py::arg("chunk_overlap") = py::none(),
            py::arg("num_workers") = py::none()
        )
        .def("_merge",
            [](lazyllm::SentenceSplitter& self, py::list splits, int chunk_size) {
                if (py::len(splits) == 0) return std::vector<std::string>{};

                struct SplitData { std::string text; bool is_sentence; int token_size; };
                std::vector<SplitData> data;
                data.reserve(py::len(splits));
                for (auto item : splits) {
                    py::object s = py::reinterpret_borrow<py::object>(item);
                    data.push_back({
                        s.attr("text").cast<std::string>(),
                        s.attr("is_sentence").cast<bool>(),
                        s.attr("token_size").cast<int>()
                    });
                }

                std::vector<std::string> chunks;
                std::vector<std::pair<std::string, int>> cur_chunk;
                int cur_chunk_len = 0;
                bool is_chunk_new = true;
                int overlap = self.overlap();

                auto close_chunk = [&]() {
                    std::string joined;
                    joined.reserve(256);
                    for (const auto& part : cur_chunk) joined += part.first;
                    chunks.push_back(joined);

                    auto last_chunk = cur_chunk;
                    cur_chunk.clear();
                    cur_chunk_len = 0;
                    is_chunk_new = true;

                    int overlap_len = 0;
                    for (auto it = last_chunk.rbegin(); it != last_chunk.rend(); ++it) {
                        if (overlap_len + it->second > overlap) break;
                        cur_chunk.push_back(*it);
                        overlap_len += it->second;
                        cur_chunk_len += it->second;
                    }
                    std::reverse(cur_chunk.begin(), cur_chunk.end());
                };

                size_t i = 0;
                while (i < data.size()) {
                    const auto& cur_split = data[i];
                    if (cur_split.token_size > chunk_size) {
                        throw py::value_error("Single token exceeded chunk size");
                    }
                    if (cur_chunk_len + cur_split.token_size > chunk_size && !is_chunk_new) {
                        close_chunk();
                    } else {
                        if (cur_split.is_sentence
                            || cur_chunk_len + cur_split.token_size <= chunk_size
                            || is_chunk_new) {
                            cur_chunk_len += cur_split.token_size;
                            cur_chunk.push_back({cur_split.text, cur_split.token_size});
                            ++i;
                            is_chunk_new = false;
                        } else {
                            close_chunk();
                        }
                    }
                }

                if (!is_chunk_new) {
                    std::string joined;
                    joined.reserve(256);
                    for (const auto& part : cur_chunk) joined += part.first;
                    chunks.push_back(joined);
                }

                std::vector<std::string> out;
                for (const auto& chunk : chunks) {
                    std::string trimmed = Trim(chunk);
                    if (!trimmed.empty()) out.push_back(trimmed);
                }
                return out;
            },
            py::arg("splits"), py::arg("chunk_size")
        );
}

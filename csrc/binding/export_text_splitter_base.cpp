#include "lazyllm.hpp"

#include "text_splitter_base.hpp"

#include <any>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

using SplitFn = lazyllm::TextSplitterBase::SplitFn;

std::any py_to_any(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) return value.cast<bool>();
    if (py::isinstance<py::int_>(value)) return value.cast<int>();
    if (py::isinstance<py::float_>(value)) return value.cast<double>();
    if (py::isinstance<py::str>(value)) return value.cast<std::string>();
    throw std::runtime_error("Unsupported default parameter type.");
}

py::object any_to_py(const std::any& value) {
    if (value.type() == typeid(bool)) return py::bool_(std::any_cast<bool>(value));
    if (value.type() == typeid(int)) return py::int_(std::any_cast<int>(value));
    if (value.type() == typeid(double)) return py::float_(std::any_cast<double>(value));
    if (value.type() == typeid(std::string)) return py::str(std::any_cast<std::string>(value));
    return py::none();
}

class PyTokenizer final : public Tokenizer {
public:
    enum class Mode {
        Generic,
        HuggingFace
    };

    explicit PyTokenizer(py::object obj, Mode mode = Mode::Generic)
        : _obj(std::move(obj)), _mode(mode) {}

    std::vector<int> encode(const std::string_view& text) const override {
        py::gil_scoped_acquire gil;
        py::object func = py::getattr(_obj, "encode", py::none());
        if (func.is_none()) throw std::runtime_error("Tokenizer missing method: encode");

        py::object result;
        if (_mode == Mode::HuggingFace) {
            result = func(std::string(text), py::arg("add_special_tokens") = false);
        } else {
            result = func(std::string(text));
        }
        return result.cast<std::vector<int>>();
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        py::gil_scoped_acquire gil;
        py::object func = py::getattr(_obj, "decode", py::none());
        if (func.is_none()) throw std::runtime_error("Tokenizer missing method: decode");

        py::object result;
        if (_mode == Mode::HuggingFace) {
            result = func(token_ids, py::arg("skip_special_tokens") = true);
        } else {
            result = func(token_ids);
        }
        return result.cast<std::string>();
    }

private:
    py::object _obj;
    Mode _mode;
};

class PyTextSplitterBase final : public lazyllm::TextSplitterBase {
public:
    using lazyllm::TextSplitterBase::TextSplitterBase;

    std::vector<lazyllm::DocNode> transform(const lazyllm::DocNode* node) const override {
        PYBIND11_OVERRIDE(
            std::vector<lazyllm::DocNode>,
            lazyllm::TextSplitterBase,
            transform,
            node
        );
    }
};

} // namespace

void exportTextSpliterBase(py::module& m) {
    py::class_<lazyllm::TextSplitterBase, lazyllm::NodeTransform, PyTextSplitterBase>(m, "_TextSplitterBase")
        .def(py::init([](
                 std::optional<unsigned> chunk_size,
                 std::optional<unsigned> overlap,
                 std::optional<unsigned> num_workers,
                 py::object encoding_name) {
                return std::make_unique<lazyllm::TextSplitterBase>(
                    chunk_size, overlap, num_workers, encoding_name.is_none() ? "gpt2" : encoding_name.cast<std::string>());
            }),
            py::arg("chunk_size") = py::none(),
            py::arg("overlap") = py::none(),
            py::arg("num_workers") = py::none(),
            py::arg("encoding_name") = py::none()
        )
        .def("split_text",
            [](const lazyllm::TextSplitterBase& self, const std::string& text, int metadata_size) {
                if (text.empty()) return std::vector<std::string>{""};
                return self.split_text(text, metadata_size);
            },
            py::arg("text"), py::arg("metadata_size")
        )
        .def_static("split_text_keep_separator",
            [](const std::string& text, const std::string& separator) {
                auto views = lazyllm::TextSplitterBase::split_text_while_keeping_separator(text, separator);
                std::vector<std::string> out;
                out.reserve(views.size());
                for (auto view : views) out.emplace_back(view);
                return out;
            },
            py::arg("text"), py::arg("separator")
        )
        .def("from_tiktoken_encoder",
            [](lazyllm::TextSplitterBase& self,
               const std::string& encoding_name,
               py::object model_name,
               py::object /*allowed_special*/,
               py::object /*disallowed_special*/,
               py::kwargs /*kwargs*/) -> lazyllm::TextSplitterBase& {
                if (model_name.is_none()) {
                    return self.from_tiktoken_encoder(encoding_name, std::nullopt);
                }
                return self.from_tiktoken_encoder(encoding_name, model_name.cast<std::string>());
            },
            py::arg("encoding_name") = "gpt2",
            py::arg("model_name") = py::none(),
            py::arg("allowed_special") = py::none(),
            py::arg("disallowed_special") = "all",
            py::return_value_policy::reference
        )
        .def("from_tiktoken_encoding",
            [](lazyllm::TextSplitterBase& self, const std::string& encoding_name) -> lazyllm::TextSplitterBase& {
                return self.from_tiktoken_encoder(encoding_name, std::nullopt);
            },
            py::arg("encoding_name") = "gpt2",
            py::return_value_policy::reference
        )
        .def("from_tokenizer",
            [](lazyllm::TextSplitterBase& self, py::object tokenizer) -> lazyllm::TextSplitterBase& {
                self.set_tokenizer(std::make_shared<PyTokenizer>(std::move(tokenizer), PyTokenizer::Mode::Generic));
                return self;
            },
            py::arg("tokenizer"),
            py::return_value_policy::reference
        )
        .def("from_huggingface_tokenizer",
            [](lazyllm::TextSplitterBase& self, py::object tokenizer) -> lazyllm::TextSplitterBase& {
                self.set_tokenizer(std::make_shared<PyTokenizer>(std::move(tokenizer), PyTokenizer::Mode::HuggingFace));
                return self;
            },
            py::arg("tokenizer"),
            py::return_value_policy::reference
        )
        .def("set_split_fns",
            [](lazyllm::TextSplitterBase& self, const std::vector<SplitFn>& split_fns, py::object sub_split_fns) {
                if (sub_split_fns.is_none()) {
                    self.set_split_functions(split_fns, std::nullopt);
                    return;
                }
                self.set_split_functions(split_fns, sub_split_fns.cast<std::vector<SplitFn>>());
            },
            py::arg("split_fns"), py::arg("sub_split_fns") = py::none()
        )
        .def("add_split_fn",
            [](lazyllm::TextSplitterBase& self, const SplitFn& split_fn, py::object index) {
                if (index.is_none()) {
                    self.add_split_function(split_fn, std::nullopt);
                    return;
                }
                self.add_split_function(split_fn, index.cast<size_t>());
            },
            py::arg("split_fn"), py::arg("index") = py::none()
        )
        .def("clear_split_fns", &lazyllm::TextSplitterBase::clear_split_functions)
        .def_static("set_default",
            [](py::kwargs kwargs) {
                lazyllm::MapParams::MapType updates;
                for (auto item : kwargs) {
                    const auto key = py::cast<std::string>(item.first);
                    updates[key] = py_to_any(item.second);
                }
                lazyllm::TextSplitterBase::_default_params.set_default(updates);
            }
        )
        .def_static("get_default",
            [](py::object param_name) -> py::object {
                const auto defaults = lazyllm::TextSplitterBase::_default_params.get_default();
                if (param_name.is_none()) {
                    py::dict out;
                    for (const auto& [key, value] : defaults) {
                        out[py::str(key)] = any_to_py(value);
                    }
                    return py::object(std::move(out));
                }

                std::string key = param_name.cast<std::string>();
                auto it = defaults.find(key);
                if (it == defaults.end() && key == "num_workers") it = defaults.find("worker_num");
                else if (it == defaults.end() && key == "worker_num") it = defaults.find("num_workers");
                if (it == defaults.end()) return py::none();
                return any_to_py(it->second);
            },
            py::arg("param_name") = py::none()
        )
        .def_static("reset_default", []() { lazyllm::TextSplitterBase::_default_params.reset_default(); });
}

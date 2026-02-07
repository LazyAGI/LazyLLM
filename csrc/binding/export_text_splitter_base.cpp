#include "lazyllm.hpp"

#include "adaptor_base_wrapper.hpp"
#include "text_spliter_base.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

class PyTokenizer final : public lazyllm::Tokenizer, public lazyllm::AdaptorBaseWrapper {
public:
    explicit PyTokenizer(const py::object& obj) : AdaptorBaseWrapper(obj) {}

    std::vector<int> encode(const std::string& text) const override {
        auto result = call("encode", {{"text", text}});
        return std::any_cast<std::vector<int>>(result);
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        auto result = call("decode", {{"token_ids", token_ids}});
        return std::any_cast<std::string>(result);
    }

private:
    std::any call_impl(
        const std::string& func_name,
        const py::object& func,
        const std::unordered_map<std::string, std::any>& args) const override
    {
        if (func.is_none()) {
            throw std::runtime_error("Tokenizer missing method: " + func_name);
        }
        if (func_name == "encode") {
            const auto& text = std::any_cast<const std::string&>(args.at("text"));
            py::object result = func(text);
            return std::any(result.cast<std::vector<int>>());
        }
        if (func_name == "decode") {
            const auto& ids = std::any_cast<const std::vector<int>&>(args.at("token_ids"));
            py::object result = func(ids);
            return std::any(result.cast<std::string>());
        }
        throw std::runtime_error("Unknown tokenizer method: " + func_name);
    }
};

} // namespace

void exportTextSpliterBase(py::module& m) {
    py::class_<lazyllm::TextSplitterBase, lazyllm::NodeTransform>(m, "TextSplitterBase")
        .def(py::init<
                std::optional<int>,
                std::optional<int>,
                std::optional<int>,
                std::optional<std::string>>(),
            py::arg("chunk_size") = py::none(),
            py::arg("overlap") = py::none(),
            py::arg("num_workers") = py::none(),
            py::arg("sentencepiece_model") = py::none()
        )
        .def("split_text", &lazyllm::TextSplitterBase::split_text,
            py::arg("text"), py::arg("metadata_size"))
        .def("from_sentencepiece_model", &lazyllm::TextSplitterBase::from_sentencepiece_model,
            py::arg("model_path"), py::return_value_policy::reference)
        .def("from_tokenizer",
            [](lazyllm::TextSplitterBase& self, py::object tokenizer) -> lazyllm::TextSplitterBase& {
                auto adaptor = std::make_shared<PyTokenizer>(tokenizer);
                self.set_tokenizer(adaptor);
                return self;
            },
            py::arg("tokenizer"),
            py::return_value_policy::reference
        )
        .def_static("set_default",
            [](py::kwargs kwargs) {
                std::unordered_map<std::string, int> params;
                for (auto item : kwargs)
                    params[py::cast<std::string>(item.first)] = py::cast<int>(item.second);
                lazyllm::TextSplitterBase::set_default(params);
            }
        )
        .def_static("get_default",
            [](py::object name) {
                if (name.is_none()) return py::cast(lazyllm::TextSplitterBase::get_default());
                auto opt = lazyllm::TextSplitterBase::get_default(name.cast<std::string>());
                if (!opt.has_value()) return py::none();
                return py::cast(*opt);
            },
            py::arg("param_name") = py::none()
        )
        .def_static("reset_default", &lazyllm::TextSplitterBase::reset_default);

    py::class_<lazyllm::_TokenTextSplitter, lazyllm::TextSplitterBase>(m, "_TokenTextSplitter")
        .def(py::init<
                std::optional<int>,
                std::optional<int>,
                std::optional<int>,
                std::optional<std::string>>(),
            py::arg("chunk_size") = py::none(),
            py::arg("overlap") = py::none(),
            py::arg("num_workers") = py::none(),
            py::arg("sentencepiece_model") = py::none()
        );

    m.def("split_text_keep_separator", &lazyllm::split_text_keep_separator,
        py::arg("text"), py::arg("separator"));
}

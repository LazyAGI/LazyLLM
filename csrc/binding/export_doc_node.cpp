#include <set>
#include <string>
#include <vector>

#include "binding_utils.hpp"
#include "doc_node.hpp"
#include "lazyllm.hpp"

namespace {

namespace pyu = lazyllm::pybind_utils;

struct PyDocNodeCore : lazyllm::DocNodeCore {
    using lazyllm::DocNodeCore::DocNodeCore;

    std::string get_metadata_string(lazyllm::MetadataMode mode) const override {
        PYBIND11_OVERRIDE(
            std::string,
            lazyllm::DocNodeCore,
            get_metadata_string,
            mode
        );
    }

    std::string get_text(lazyllm::MetadataMode mode) const override {
        PYBIND11_OVERRIDE(
            std::string,
            lazyllm::DocNodeCore,
            get_text,
            mode
        );
    }
};

lazyllm::DocNodeCore::Metadata MetadataFromPy(const py::object& obj) {
    lazyllm::DocNodeCore::Metadata out;
    if (obj.is_none()) return out;
    py::dict d = py::dict(obj);
    out.reserve(d.size());
    for (auto item : d) {
        const std::string key = py::cast<std::string>(item.first);
        out.emplace(key, pyu::PyToMetadataValue(item.second));
    }
    return out;
}

std::set<std::string> StringSetFromPy(const py::object& obj) {
    std::set<std::string> keys;
    if (obj.is_none()) return keys;
    for (auto item : obj) keys.insert(py::str(item).cast<std::string>());
    return keys;
}

lazyllm::MetadataMode ParseMode(const py::object& mode, lazyllm::MetadataMode default_mode) {
    if (mode.is_none()) return default_mode;
    return pyu::ParseMetadataMode(mode);
}

} // namespace

void exportDocNode(py::module& m) {
    py::enum_<lazyllm::MetadataMode>(m, "MetadataMode")
        .value("ALL", lazyllm::MetadataMode::ALL)
        .value("EMBED", lazyllm::MetadataMode::EMBED)
        .value("LLM", lazyllm::MetadataMode::LLM)
        .value("NONE", lazyllm::MetadataMode::NONE);

    py::class_<lazyllm::DocNodeCore, PyDocNodeCore, std::shared_ptr<lazyllm::DocNodeCore>>(
        m, "DocNodeCore", py::dynamic_attr()
    )
        .def(py::init([](const py::object& text, const py::object& metadata, const py::object& uid) {
            return std::make_shared<PyDocNodeCore>(
                text.is_none() ? std::string() : py::str(text).cast<std::string>(),
                MetadataFromPy(metadata),
                uid.is_none() ? std::string() : py::cast<std::string>(uid)
            );
        }),
            py::arg("text") = py::none(),
            py::arg("metadata") = py::none(),
            py::arg("uid") = py::none()
        )
        .def_readwrite("_uid", &lazyllm::DocNodeCore::_uid)
        .def_readwrite("_text", &lazyllm::DocNodeCore::_text)
        .def_readwrite("_metadata", &lazyllm::DocNodeCore::_metadata)
        .def_property("_excluded_embed_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                return std::vector<std::string>(
                    node._excluded_embed_metadata_keys.begin(),
                    node._excluded_embed_metadata_keys.end()
                );
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node._excluded_embed_metadata_keys = StringSetFromPy(keys_obj);
            }
        )
        .def_property("_excluded_llm_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                return std::vector<std::string>(
                    node._excluded_llm_metadata_keys.begin(),
                    node._excluded_llm_metadata_keys.end()
                );
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node._excluded_llm_metadata_keys = StringSetFromPy(keys_obj);
            }
        )
        .def("get_metadata_str", [](const lazyllm::DocNodeCore& node, const py::object& mode) {
            return node.get_metadata_string(ParseMode(mode, lazyllm::MetadataMode::ALL));
        }, py::arg("mode") = py::none())
        .def("get_text", [](const lazyllm::DocNodeCore& node, const py::object& metadata_mode) {
            return node.get_text(ParseMode(metadata_mode, lazyllm::MetadataMode::NONE));
        }, py::arg("metadata_mode") = py::none());
}

#include <set>
#include <string>
#include <vector>

#include "binding_utils.hpp"
#include "doc_node.hpp"
#include "lazyllm.hpp"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(lazyllm::DocNodeCore::Metadata);

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
};

lazyllm::DocNodeCore::Metadata MetadataFromPy(const py::object& obj) {
    lazyllm::DocNodeCore::Metadata out;
    if (obj.is_none() || !py::isinstance<py::dict>(obj)) return out;
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

    auto metadata_cls = py::bind_map<lazyllm::DocNodeCore::Metadata>(m, "MetadataMap");
    metadata_cls
        .def("get",
            [](const lazyllm::DocNodeCore::Metadata& self, const std::string& key, const py::object& default_value) {
                auto it = self.find(key);
                if (it == self.end()) return default_value;
                return py::cast(it->second);
            },
            py::arg("key"), py::arg("default") = py::none()
        )
        .def("pop",
            [](lazyllm::DocNodeCore::Metadata& self, const std::string& key) {
                auto it = self.find(key);
                if (it == self.end()) throw py::key_error(key);
                py::object value = py::cast(it->second);
                self.erase(it);
                return value;
            },
            py::arg("key")
        )
        .def("pop",
            [](lazyllm::DocNodeCore::Metadata& self, const std::string& key, const py::object& default_value) {
                auto it = self.find(key);
                if (it == self.end()) return default_value;
                py::object value = py::cast(it->second);
                self.erase(it);
                return value;
            },
            py::arg("key"), py::arg("default")
        )
        .def("copy",
            [](const lazyllm::DocNodeCore::Metadata& self) {
                py::dict out;
                for (const auto& [k, v] : self) out[py::str(k)] = py::cast(v);
                return out;
            }
        )
        .def("update",
            [](lazyllm::DocNodeCore::Metadata& self, const py::object& other) {
                py::dict d = py::dict(other);
                for (auto item : d) {
                    const std::string key = py::cast<std::string>(item.first);
                    self[key] = pyu::PyToMetadataValue(item.second);
                }
            },
            py::arg("other")
        )
        .def("__eq__",
            [](const lazyllm::DocNodeCore::Metadata& self, const py::object& other) {
                py::dict out;
                for (const auto& [k, v] : self) out[py::str(k)] = py::cast(v);
                const int cmp = PyObject_RichCompareBool(out.ptr(), other.ptr(), Py_EQ);
                if (cmp < 0) throw py::error_already_set();
                return cmp == 1;
            },
            py::is_operator()
        )
        .def("__repr__",
            [](const lazyllm::DocNodeCore::Metadata& self) {
                py::dict out;
                for (const auto& [k, v] : self) out[py::str(k)] = py::cast(v);
                return py::repr(out).cast<std::string>();
            }
        )
        .def("__deepcopy__",
            [](const lazyllm::DocNodeCore::Metadata& self, const py::dict&) {
                py::dict out;
                for (const auto& [k, v] : self) out[py::str(k)] = py::cast(v);
                return out;
            },
            py::arg("memo")
        );

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
        .def_readonly("_uid", &lazyllm::DocNodeCore::_uid)
        .def_readwrite("_text", &lazyllm::DocNodeCore::_text)
        .def_property_readonly("uid",
            [](const lazyllm::DocNodeCore& node) -> const std::string& {
                return node._uid;
            }
        )
        .def_property("_metadata",
            [](lazyllm::DocNodeCore& node) -> lazyllm::DocNodeCore::Metadata& {
                return node._metadata;
            },
            [](lazyllm::DocNodeCore& node, const py::object& metadata) {
                node._metadata = MetadataFromPy(metadata);
            },
            py::return_value_policy::reference_internal
        )
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
        }, py::arg("mode") = py::none());
}

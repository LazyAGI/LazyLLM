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

py::dict MetadataToPyDict(const lazyllm::DocNodeCore::Metadata& self) {
    py::dict out;
    for (const auto& [k, v] : self) out[py::str(k)] = py::cast(v);
    return out;
}

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

py::object CloneDocNodeCore(py::object self_obj) {
    auto self = self_obj.cast<std::shared_ptr<lazyllm::DocNodeCore>>();
    auto copy = std::make_shared<PyDocNodeCore>(
        self->_text, self->_metadata, self->_uid
    );
    copy->_excluded_embed_metadata_keys = self->_excluded_embed_metadata_keys;
    copy->_excluded_llm_metadata_keys = self->_excluded_llm_metadata_keys;
    py::object copy_obj = py::cast(copy);
    if (py::hasattr(self_obj, "__dict__")) {
        py::dict src_dict = self_obj.attr("__dict__");
        py::dict dst_dict = copy_obj.attr("__dict__");
        for (auto item : src_dict) {
            dst_dict[item.first] = item.second;
        }
    }
    if (py::hasattr(self_obj, "__class__")) {
        copy_obj.attr("__class__") = self_obj.attr("__class__");
    }
    return copy_obj;
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
            [](const lazyllm::DocNodeCore::Metadata& self, const std::string& key, const py::object& default_value
            ) -> py::object {
                auto it = self.find(key);
                if (it == self.end()) return default_value;
                return py::cast(it->second);
            },
            py::arg("key"), py::arg("default") = py::none()
        )
        .def("pop",
            [](lazyllm::DocNodeCore::Metadata& self, const std::string& key) -> py::object {
                auto it = self.find(key);
                if (it == self.end()) throw py::key_error(key);
                py::object value = py::cast(it->second);
                self.erase(it);
                return value;
            },
            py::arg("key")
        )
        .def("pop",
            [](lazyllm::DocNodeCore::Metadata& self, const std::string& key, const py::object& default_value
            ) -> py::object {
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
                return MetadataToPyDict(self);
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
            [](const lazyllm::DocNodeCore::Metadata& self, const py::object& other) -> py::object {
                const int cmp = PyObject_RichCompareBool(MetadataToPyDict(self).ptr(), other.ptr(), Py_EQ);
                if (cmp < 0) {
                    PyErr_Clear();
                    Py_INCREF(Py_NotImplemented);
                    return py::reinterpret_steal<py::object>(Py_NotImplemented);
                }
                return py::bool_(cmp == 1);
            },
            py::is_operator()
        )
        .def("__repr__",
            [](const lazyllm::DocNodeCore::Metadata& self) {
                return py::repr(MetadataToPyDict(self)).cast<std::string>();
            }
        )
        .def("__deepcopy__",
            [](const lazyllm::DocNodeCore::Metadata& self, const py::dict&) {
                return MetadataToPyDict(self);
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
        .def_readwrite("_uid", &lazyllm::DocNodeCore::_uid)
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
        }, py::arg("mode") = py::none())
        .def("__copy__", [](py::object self_obj) {
            return CloneDocNodeCore(self_obj);
        })
        .def("__deepcopy__", [](py::object self_obj, const py::dict&) {
            return CloneDocNodeCore(self_obj);
        }, py::arg("memo"));
}

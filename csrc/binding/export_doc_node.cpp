#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "lazyllm.hpp"
#include "doc_node.hpp"
#include "binding_utils.hpp"
#include "map_binding_helper.hpp"
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(lazyllm::DocNodeCore::Metadata);

namespace {

namespace pyu = lazyllm::pybind_utils;

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

py::dict MetadataToPy(const lazyllm::DocNodeCore::Metadata& meta) {
    py::dict d;
    for (const auto& [key, value] : meta) {
        d[py::str(key)] = pyu::MetadataValueToPy(value);
    }
    return d;
}

std::set<std::string> StringSetFromPy(const py::object& obj) {
    std::set<std::string> keys;
    if (obj.is_none()) return keys;
    for (auto item : obj) {
        keys.insert(py::cast<std::string>(item));
    }
    return keys;
}

std::string ResolveTextFromPy(const py::object& self) {
    std::string class_name = "DocNodeCore";
    try {
        class_name = py::cast<std::string>(py::type::of(self).attr("__name__"));
    } catch (const py::error_already_set&) {
    }

    if (class_name != "DocNodeCore") {
        try {
            py::object text = py::getattr(self, "text", py::none());
            if (!text.is_none()) {
                if (py::isinstance<py::str>(text)) return text.cast<std::string>();
                std::vector<std::string> texts;
                if (pyu::ExtractStringSequence(text, &texts)) return JoinLines(texts);
                return py::str(text).cast<std::string>();
            }
        } catch (const py::error_already_set&) {
        }
    }

    py::object raw_text = py::getattr(self, "_text", py::none());
    if (!raw_text.is_none()) {
        if (py::isinstance<py::str>(raw_text)) return raw_text.cast<std::string>();
        return py::str(raw_text).cast<std::string>();
    }
    return std::string();
}

std::string BuildCoreMetadataString(const py::object& self, lazyllm::MetadataMode mode) {
    if (mode == lazyllm::MetadataMode::NONE) return "";

    py::object metadata_obj = py::getattr(self, "metadata", py::none());
    if (metadata_obj.is_none()) metadata_obj = py::getattr(self, "_metadata", py::none());
    const auto metadata_map = MetadataFromPy(metadata_obj);

    std::set<std::string> valid_keys;
    for (const auto& [key, _] : metadata_map) valid_keys.insert(key);

    if (mode == lazyllm::MetadataMode::LLM) {
        py::object keys_obj = py::getattr(self, "excluded_llm_metadata_keys", py::none());
        if (keys_obj.is_none()) keys_obj = py::getattr(self, "_excluded_llm_metadata_keys", py::none());
        valid_keys = SetDiff(valid_keys, StringSetFromPy(keys_obj));
    } else if (mode == lazyllm::MetadataMode::EMBED) {
        py::object keys_obj = py::getattr(self, "excluded_embed_metadata_keys", py::none());
        if (keys_obj.is_none()) keys_obj = py::getattr(self, "_excluded_embed_metadata_keys", py::none());
        valid_keys = SetDiff(valid_keys, StringSetFromPy(keys_obj));
    }

    std::vector<std::string> kv_strings;
    for (const std::string& key : valid_keys) {
        kv_strings.emplace_back(key + ": " + any_to_string(metadata_map.at(key)));
    }
    return JoinLines(kv_strings);
}

std::string BuildCoreText(const py::object& self, lazyllm::MetadataMode mode) {
    const std::string text = ResolveTextFromPy(self);
    if (mode == lazyllm::MetadataMode::NONE) return text;
    const auto metadata_string = BuildCoreMetadataString(self, mode);
    if (metadata_string.empty()) return text;
    return metadata_string + "\n\n" + text;
}

lazyllm::DocNodeCore init_core(py::object text, py::object metadata, std::optional<std::string> uid) {
    const auto metadata_map = MetadataFromPy(metadata);
    lazyllm::DocNodeCore node("", uid.value_or(""), metadata_map);
    if (!text.is_none()) node.set_root_text(text.cast<std::string>());
    return node;
}

} // namespace

void exportDocNode(py::module& m) {
    py::enum_<lazyllm::MetadataMode>(m, "MetadataMode")
        .value("ALL", lazyllm::MetadataMode::ALL)
        .value("EMBED", lazyllm::MetadataMode::EMBED)
        .value("LLM", lazyllm::MetadataMode::LLM)
        .value("NONE", lazyllm::MetadataMode::NONE);

    auto metadata_map = py::bind_map<lazyllm::DocNodeCore::Metadata>(m, "DocNodeMetadataMap");
    pyu::RegisterMapAsMutableMapping(metadata_map);
    pyu::BindDictLikeMethods<lazyllm::DocNodeCore::Metadata>(
        metadata_map,
        [](const lazyllm::DocNodeCore::MetadataVType& value) { return pyu::MetadataValueToPy(value); },
        [](py::object value) { return pyu::PyToMetadataValue(value); },
        [](const lazyllm::DocNodeCore::Metadata& self) { return MetadataToPy(self); }
    );

    py::class_<lazyllm::DocNodeCore, std::shared_ptr<lazyllm::DocNodeCore>>(m, "DocNodeCore", py::dynamic_attr())
        .def(py::init(&init_core),
            py::arg("text") = py::none(),
            py::arg("metadata") = py::none(),
            py::arg("uid") = py::none()
        )
        .def_property("_uid",
            [](const lazyllm::DocNodeCore& node) { return node.get_uid(); },
            [](lazyllm::DocNodeCore& node, const std::string& value) { node.set_uid(value); }
        )
        .def_property_readonly("uid", [](const lazyllm::DocNodeCore& node) { return node.get_uid(); })
        .def_property("_text",
            [](const lazyllm::DocNodeCore& node) { return std::string(node.get_text(lazyllm::MetadataMode::NONE)); },
            [](lazyllm::DocNodeCore& node, const std::string& value) { node.set_root_text(value); }
        )
        .def_property_readonly("text",
            py::cpp_function([](py::object self) { return ResolveTextFromPy(self); })
        )
        .def_property("_metadata",
            py::cpp_function([](lazyllm::DocNodeCore& node) -> lazyllm::DocNodeCore::Metadata& {
                return node._metadata;
            }, py::return_value_policy::reference_internal),
            [](lazyllm::DocNodeCore& node, const py::object& meta) {
                node._metadata = MetadataFromPy(meta);
            }
        )
        .def_property("metadata",
            py::cpp_function([](lazyllm::DocNodeCore& node) -> lazyllm::DocNodeCore::Metadata& {
                return node._metadata;
            }, py::return_value_policy::reference_internal),
            [](lazyllm::DocNodeCore& node, const py::object& meta) {
                node._metadata = MetadataFromPy(meta);
            }
        )
        .def_property("_excluded_embed_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                const auto keys = node.get_excluded_embed_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node.set_excluded_embed_metadata_keys(StringSetFromPy(keys_obj));
            }
        )
        .def_property("_excluded_llm_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                const auto keys = node.get_excluded_llm_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node.set_excluded_llm_metadata_keys(StringSetFromPy(keys_obj));
            }
        )
        .def_property("excluded_embed_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                const auto keys = node.get_excluded_embed_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node.set_excluded_embed_metadata_keys(StringSetFromPy(keys_obj));
            }
        )
        .def_property("excluded_llm_metadata_keys",
            [](const lazyllm::DocNodeCore& node) {
                const auto keys = node.get_excluded_llm_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNodeCore& node, const py::object& keys_obj) {
                node.set_excluded_llm_metadata_keys(StringSetFromPy(keys_obj));
            }
        )
        .def("get_metadata_str", [](py::object self, const py::object& mode) {
            if (mode.is_none()) return BuildCoreMetadataString(self, lazyllm::MetadataMode::ALL);
            return BuildCoreMetadataString(self, pyu::ParseMetadataMode(mode));
        }, py::arg("mode") = py::none())
        .def("get_text", [](py::object self, const py::object& metadata_mode) {
            return BuildCoreText(self, pyu::ParseMetadataMode(metadata_mode));
        }, py::arg("metadata_mode") = py::none());
}

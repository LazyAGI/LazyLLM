#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "lazyllm.hpp"
#include "document_store.hpp"
#include "doc_node.hpp"
#include "binding_utils.hpp"
#include "map_binding_helper.hpp"
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(lazyllm::DocNode::Metadata);
PYBIND11_MAKE_OPAQUE(lazyllm::DocNode::Children);

namespace {

namespace pyu = lazyllm::pybind_utils;

const std::unordered_map<std::string, const char*> kDocNodeAttrAliases = {
    {"uid", "_uid"},
    {"group", "_group"},
    {"content", "_content"},
    {"parent", "_parent"},
    {"children", "_children"},
    {"global_metadata", "_global_metadata"},
    {"metadata", "_metadata"},
    {"excluded_embed_metadata_keys", "_excluded_embed_metadata_keys"},
    {"excluded_llm_metadata_keys", "_excluded_llm_metadata_keys"},
};

const std::unordered_set<std::string> kDocNodeReadonlyAliases = {
    "uid",
    "group",
};

const char* ResolveDocNodeAlias(const std::string& attr_name) {
    const auto it = kDocNodeAttrAliases.find(attr_name);
    if (it == kDocNodeAttrAliases.end()) return nullptr;
    return it->second;
}

bool IsJsonDocNode(const py::object& self) {
    try {
        const auto name = py::cast<std::string>(py::type::of(self).attr("__name__"));
        return name == "JsonDocNode";
    } catch (const py::error_already_set&) {
        return false;
    }
}

lazyllm::DocNode::Metadata MetadataFromPy(const py::object& obj) {
    lazyllm::DocNode::Metadata out;
    if (obj.is_none()) return out;
    py::dict d = py::dict(obj);
    out.reserve(d.size());
    for (auto item : d) {
        const std::string key = py::cast<std::string>(item.first);
        out.emplace(key, pyu::PyToMetadataValue(item.second));
    }
    return out;
}

py::dict MetadataToPy(const lazyllm::DocNode::Metadata& meta) {
    py::dict d;
    for (const auto& [key, value] : meta) {
        d[py::str(key)] = pyu::MetadataValueToPy(value);
    }
    return d;
}

lazyllm::DocNode::Metadata& EnsureRootGlobalMetadata(lazyllm::DocNode& node) {
    auto* root = const_cast<lazyllm::DocNode*>(node.get_root_node());
    if (!root->_p_global_metadata) {
        root->_p_global_metadata = std::make_shared<lazyllm::DocNode::Metadata>();
    }
    return *(root->_p_global_metadata);
}

const lazyllm::DocNode::Metadata& GetRootGlobalMetadata(const lazyllm::DocNode& node) {
    static const lazyllm::DocNode::Metadata empty;
    auto* root = node.get_root_node();
    if (!root || !root->_p_global_metadata) return empty;
    return *(root->_p_global_metadata);
}

lazyllm::DocNode::Children ChildrenFromPy(const py::object& obj) {
    lazyllm::DocNode::Children out;
    if (obj.is_none()) return out;
    py::dict d = py::dict(obj);
    out.reserve(d.size());
    for (auto item : d) {
        const std::string group = py::cast<std::string>(item.first);
        py::object nodes_obj = py::reinterpret_borrow<py::object>(item.second);
        std::vector<lazyllm::PDocNode> nodes;
        for (auto child_obj : nodes_obj) {
            py::object child = py::reinterpret_borrow<py::object>(child_obj);
            if (child.is_none()) continue;
            nodes.emplace_back(child.cast<lazyllm::PDocNode>());
        }
        out.emplace(group, std::move(nodes));
    }
    return out;
}

py::dict ChildrenToPy(const lazyllm::DocNode::Children& children) {
    py::dict d;
    for (const auto& [group, nodes] : children) {
        py::list py_nodes;
        for (const auto& n : nodes) py_nodes.append(n);
        d[py::str(group)] = std::move(py_nodes);
    }
    return d;
}

std::vector<lazyllm::PDocNode> NodeVectorFromPy(const py::object& obj) {
    std::vector<lazyllm::PDocNode> out;
    if (obj.is_none()) return out;
    for (auto item : obj) {
        py::object node = py::reinterpret_borrow<py::object>(item);
        if (node.is_none()) continue;
        out.emplace_back(node.cast<lazyllm::PDocNode>());
    }
    return out;
}

using NodeGroups = std::unordered_map<std::string, std::unordered_map<std::string, std::any>>;

std::optional<NodeGroups> NodeGroupsFromPy(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    py::dict d = py::dict(obj);
    NodeGroups out;
    out.reserve(d.size());
    for (auto item : d) {
        const std::string key = py::cast<std::string>(item.first);
        py::object group_obj = py::reinterpret_borrow<py::object>(item.second);
        py::dict group_dict = py::dict(group_obj);
        std::unordered_map<std::string, std::any> inner;
        inner.reserve(group_dict.size());
        for (auto kv : group_dict) {
            const std::string inner_key = py::cast<std::string>(kv.first);
            inner.emplace(inner_key, pyu::PyToAny(kv.second));
        }
        out.emplace(key, std::move(inner));
    }
    return out;
}

std::optional<std::variant<std::string, std::vector<std::string>>> NormalizeContent(
    const py::object& content
) {
    if (content.is_none()) return std::nullopt;
    if (py::isinstance<py::str>(content)) return content.cast<std::string>();
    std::vector<std::string> texts;
    if (pyu::ExtractStringSequence(content, &texts)) return texts;
    return pyu::DumpJson(content);
}

lazyllm::DocNode init(
    std::optional<std::string> uid,
    py::object content,
    std::optional<std::string> group,
    std::optional<lazyllm::DocNode::EmbeddingVecs> embedding,
    py::object parent,
    py::object store,
    py::object node_groups,
    py::object metadata,
    py::object global_metadata,
    py::object text
) {
    if (!content.is_none() && !text.is_none())
        throw std::invalid_argument("`text` and `content` cannot be set at the same time.");

    lazyllm::DocNode* p_parent_node = nullptr;
    std::shared_ptr<lazyllm::DocumentStore> store_adaptor = nullptr;

    // Build node groups map.
    // Usually, parent + store + node_groups are not None at the same time.
    const auto node_groups_opt = NodeGroupsFromPy(node_groups);
    const auto metadata_map = MetadataFromPy(metadata);
    const auto global_metadata_map = MetadataFromPy(global_metadata);
    const bool has_parent = !parent.is_none();
    const bool has_store_context = has_parent && !store.is_none()
        && node_groups_opt.has_value() && !global_metadata.is_none() && group.has_value();
    if (has_store_context) {
        std::unordered_map<std::string, lazyllm::NodeGroup> node_groups_map;
        node_groups_map.reserve(node_groups_opt->size());
        for (const auto& [group_key, group_dict] : *node_groups_opt) {
            node_groups_map.emplace(group_key, lazyllm::NodeGroup(
                std::any_cast<std::string>(group_dict.at(std::string("parent"))),
                std::any_cast<std::string>(group_dict.at(std::string("display_name")))
            ));
        }
        store_adaptor = lazyllm::DocumentStore::from_store(store, node_groups_map);

        auto kb_id = std::get<std::string>(global_metadata_map.at(
            std::string(lazyllm::RAGMetadataKeys::KB_ID)));
        auto doc_id = std::get<std::string>(global_metadata_map.at(
            std::string(lazyllm::RAGMetadataKeys::DOC_ID)));

        if (py::isinstance<py::str>(parent)) {
            const auto parent_uid = parent.cast<std::string>();
            p_parent_node = std::any_cast<lazyllm::DocNode*>(store_adaptor->call("get_node",
                {{"group_name", *group}, {"uid", parent_uid}, {"kb_id", kb_id}}));
        } else {
            p_parent_node = parent.cast<lazyllm::DocNode*>();
        }
    } else if (has_parent && !py::isinstance<py::str>(parent)) {
        p_parent_node = parent.cast<lazyllm::DocNode*>();
    }

    std::string raw_text;

    lazyllm::DocNode node(
        "",
        group.value_or(""),
        uid.value_or(""),
        p_parent_node,
        metadata_map,
        std::make_shared<lazyllm::DocNode::Metadata>(global_metadata_map)
    );
    if (store_adaptor) node.set_store(store_adaptor);
    if (embedding) {
        for (const auto& [key, vec] : *embedding)
            node.set_embedding_vec(key, vec);
    }
    if (!content.is_none()) {
        const auto normalized = NormalizeContent(content);
        if (normalized) {
            if (const auto* s = std::get_if<std::string>(&*normalized))
                node.set_root_text(std::move(*s));
            else
                node.set_root_texts(std::get<std::vector<std::string>>(*normalized));
        }
    }
    else if (!text.is_none()){
        node.set_root_text(text.cast<std::string>());
    }

    return node;
}

std::string DocNodeToString(const lazyllm::DocNode& node) {
    py::dict d;
    const auto children = node.get_children();
    for (const auto& [group, nodes] : children) {
        py::list ids;
        for (std::shared_ptr<lazyllm::DocNode> n : nodes) {
            if (n) ids.append(n->get_uid());
        }
        d[py::str(group)] = std::move(ids);
    }
    const std::string children_str = py::str(d).cast<std::string>();
    return "DocNode(id: " + node.get_uid() + ", group: " + node._group_name
        + ", content: " + node.get_text(lazyllm::MetadataMode::NONE)
        + ") parent: " + node.get_parent_uid() + ", children: " + children_str;
}

} // namespace

void exportDocNode(py::module& m) {
    py::enum_<lazyllm::MetadataMode>(m, "MetadataMode")
        .value("ALL", lazyllm::MetadataMode::ALL)
        .value("EMBED", lazyllm::MetadataMode::EMBED)
        .value("LLM", lazyllm::MetadataMode::LLM)
        .value("NONE", lazyllm::MetadataMode::NONE);

    auto metadata_map = py::bind_map<lazyllm::DocNode::Metadata>(m, "DocNodeMetadataMap");
    pyu::RegisterMapAsMutableMapping(metadata_map);
    pyu::BindDictLikeMethods<lazyllm::DocNode::Metadata>(
        metadata_map,
        [](const lazyllm::DocNode::MetadataVType& value) { return pyu::MetadataValueToPy(value); },
        [](py::object value) { return pyu::PyToMetadataValue(value); },
        [](const lazyllm::DocNode::Metadata& self) { return MetadataToPy(self); }
    );

    auto children_map = py::bind_map<lazyllm::DocNode::Children>(m, "DocNodeChildrenMap");
    pyu::RegisterMapAsMutableMapping(children_map);
    pyu::BindDictLikeMethods<lazyllm::DocNode::Children>(
        children_map,
        [](const std::vector<lazyllm::PDocNode>& value) -> py::object { return py::cast(value); },
        [](py::object value) { return NodeVectorFromPy(value); },
        [](const lazyllm::DocNode::Children& self) { return ChildrenToPy(self); },
        py::list()
    );

    py::class_<lazyllm::DocNode, std::shared_ptr<lazyllm::DocNode>>(m, "DocNode", py::dynamic_attr())
        .def(py::init(&init),
            py::arg("uid") = py::none(),
            py::arg("content") = py::none(),
            py::arg("group") = py::none(),
            py::arg("embedding") = py::none(),
            py::arg("parent") = py::none(),
            py::arg("store") = py::none(),
            py::arg("node_groups") = py::none(),
            py::arg("metadata") = py::none(),
            py::arg("global_metadata") = py::none(),
            py::arg("text") = py::none()
        )
        .def_property("_uid",
            [](const lazyllm::DocNode& node) { return node.get_uid(); },
            [](lazyllm::DocNode& node, const std::string& value) { node.set_uid(value); }
        )
        .def_property("_group",
            [](const lazyllm::DocNode& node) { return node._group_name; },
            [](lazyllm::DocNode& node, const std::string& value) { node._group_name = value; }
        )
        .def_property("_content",
            py::cpp_function([](const py::object& self) {
                const auto& node = self.cast<const lazyllm::DocNode&>();
                const std::string text = std::string(node.get_text(lazyllm::MetadataMode::NONE));
                if (!IsJsonDocNode(self)) return py::cast(text);
                try {
                    return pyu::LoadJson(text);
                } catch (const py::error_already_set&) {
                    return py::cast(text);
                }
            }),
            py::cpp_function([](py::object self, const py::object& content) {
                auto& node = self.cast<lazyllm::DocNode&>();
                const auto normalized = NormalizeContent(content);
                if (!normalized) return;
                if (const auto* content_str = std::get_if<std::string>(&*normalized)) {
                    node.set_root_text(std::move(*content_str));
                    return;
                }
                node.set_root_texts(std::get<std::vector<std::string>>(*normalized));
            })
        )
        .def_property("number",
            [](const lazyllm::DocNode& node) {
                auto it = node._metadata.find("lazyllm_store_num");
                if (it == node._metadata.end()) return 0;
                if (auto* value = std::get_if<int>(&it->second)) return *value;
                if (auto* value = std::get_if<double>(&it->second)) return static_cast<int>(*value);
                return 0;
            },
            [](lazyllm::DocNode& node, int value) {
                node._metadata[std::string("lazyllm_store_num")] = value;
            }
        )
        .def_property_readonly("text", [](const lazyllm::DocNode& node) { return std::string(node.get_text()); })
        .def_property_readonly("content_hash", [](const lazyllm::DocNode& node) {
            return lazyllm::to_hex(node.get_text_hash());
        })
        .def_property("embedding",
            [](const lazyllm::DocNode& node) { return node._embedding_vecs; },
            [](lazyllm::DocNode& node, const lazyllm::DocNode::EmbeddingVecs& v) {
                node._embedding_vecs = v;
            }
        )
        .def_property("_parent",
            [](const lazyllm::DocNode& node) { return node.get_parent_node(); },
            [](lazyllm::DocNode& node, const py::object& parent) {
                if (parent.is_none() || py::isinstance<py::str>(parent)) {
                    node.set_parent_node(nullptr);
                    return;
                }
                node.set_parent_node(parent.cast<lazyllm::DocNode*>());
            },
            py::return_value_policy::reference
        )
        .def_property("_children",
            py::cpp_function([](lazyllm::DocNode& node) -> lazyllm::DocNode::Children& {
                node.get_children();
                return node.get_children_ref();
            }, py::return_value_policy::reference_internal),
            [](lazyllm::DocNode& node, const py::object& children) {
                node.set_children(ChildrenFromPy(children));
            }
        )
        .def_property_readonly("root_node",
            [](const lazyllm::DocNode& node) { return node.get_root_node(); },
            py::return_value_policy::reference
        )
        .def_property_readonly("is_root_node",
            [](const lazyllm::DocNode& node) { return node.get_parent_node() == nullptr; }
        )
        .def_property("_global_metadata",
            py::cpp_function([](lazyllm::DocNode& node) -> lazyllm::DocNode::Metadata& {
                return EnsureRootGlobalMetadata(node);
            }, py::return_value_policy::reference_internal),
            [](lazyllm::DocNode& node, const py::object& meta) {
                EnsureRootGlobalMetadata(node) = MetadataFromPy(meta);
            }
        )
        .def_property("_metadata",
            py::cpp_function([](lazyllm::DocNode& node) -> lazyllm::DocNode::Metadata& {
                return node._metadata;
            }, py::return_value_policy::reference_internal),
            [](lazyllm::DocNode& node, const py::object& meta) {
                node._metadata = MetadataFromPy(meta);
            }
        )
        .def_property("_excluded_embed_metadata_keys",
            [](const lazyllm::DocNode& node) {
                const auto keys = node.get_excluded_embed_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNode& node, const py::object& keys_obj) {
                std::set<std::string> keys;
                for (auto item : keys_obj) {
                    keys.insert(py::cast<std::string>(item));
                }
                node.set_excluded_embed_metadata_keys(keys);
            }
        )
        .def_property("_excluded_llm_metadata_keys",
            [](const lazyllm::DocNode& node) {
                const auto keys = node.get_excluded_llm_metadata_keys();
                return std::vector<std::string>(keys.begin(), keys.end());
            },
            [](lazyllm::DocNode& node, const py::object& keys_obj) {
                std::set<std::string> keys;
                for (auto item : keys_obj) {
                    keys.insert(py::cast<std::string>(item));
                }
                node.set_excluded_llm_metadata_keys(keys);
            }
        )
        .def("__getattr__", [](py::object self, const std::string& attr_name) -> py::object {
            if (const char* alias = ResolveDocNodeAlias(attr_name)) {
                return py::getattr(self, py::str(alias));
            }
            throw py::attribute_error("DocNode has no attribute '" + attr_name + "'");
        })
        .def("__setattr__", [](py::object self, const std::string& attr_name, const py::object& value) {
            if (kDocNodeReadonlyAliases.find(attr_name) != kDocNodeReadonlyAliases.end()) {
                throw py::attribute_error("property '" + attr_name + "' of 'DocNode' object has no setter");
            }
            const char* alias = ResolveDocNodeAlias(attr_name);
            const char* target = alias ? alias : attr_name.c_str();
            py::object object_setattr = py::module_::import("builtins").attr("object").attr("__setattr__");
            object_setattr(self, py::str(target), value);
        })
        .def_property("docpath",
            [](const lazyllm::DocNode& node) { return node.get_doc_path(); },
            [](lazyllm::DocNode& node, const std::string& path) { node.set_doc_path(path); }
        )
        .def_property("embedding_state",
            [](const lazyllm::DocNode& node) { return node._pending_embedding_keys; },
            [](lazyllm::DocNode& node, const std::set<std::string>& keys) {
                node._pending_embedding_keys = keys;
            }
        )
        .def_property("relevance_score",
            [](const lazyllm::DocNode& node) { return node._relevance_score; },
            [](lazyllm::DocNode& node, double score) { node._relevance_score = score; }
        )
        .def_property("similarity_score",
            [](const lazyllm::DocNode& node) { return node._similarity_score; },
            [](lazyllm::DocNode& node, double score) { node._similarity_score = score; }
        )
        .def("get_children_str", [](const lazyllm::DocNode& node) {
            py::dict d;
            const auto children = node.get_children();
            for (const auto& [group, nodes] : children) {
                py::list ids;
                for (std::shared_ptr<lazyllm::DocNode> n : nodes) if (n) ids.append(n->get_uid());
                d[py::str(group)] = std::move(ids);
            }
            return py::str(d);
        })
        .def("get_parent_id", &lazyllm::DocNode::get_parent_uid)
        .def("__str__", &DocNodeToString)
        .def("__repr__", [](const lazyllm::DocNode& node) {
            py::object cfg = py::module_::import("lazyllm").attr("config");
            py::object mode = py::module_::import("lazyllm").attr("Mode");
            py::object cfg_mode = cfg.attr("__getitem__")("mode");
            if (py::bool_(cfg_mode.equal(mode.attr("Debug"))))
                return DocNodeToString(node);
            return std::string("<Node id=") + node.get_uid() + ">";
        })
        .def("__eq__", &lazyllm::DocNode::operator==, py::is_operator())
        .def("__hash__", [](const lazyllm::DocNode& node) {
            return static_cast<py::ssize_t>(std::hash<std::string>{}(node.get_uid()));
        })
        .def("__getstate__", [](const lazyllm::DocNode& node) {
            py::dict st;
            st["_uid"] = node.get_uid();
            st["_content"] = node.get_text(lazyllm::MetadataMode::NONE);
            st["_group"] = node._group_name;
            st["_embedding"] = node._embedding_vecs;
            st["_metadata"] = MetadataToPy(node._metadata);
            st["_global_metadata"] = MetadataToPy(GetRootGlobalMetadata(node));
            st["_excluded_embed_metadata_keys"] = node.get_excluded_embed_metadata_keys();
            st["_excluded_llm_metadata_keys"] = node.get_excluded_llm_metadata_keys();
            st["_store"] = py::none();
            st["_node_groups"] = py::none();
            return st;
        })
        .def("has_missing_embedding", [](const lazyllm::DocNode& node, const py::object& keys) {
            if (py::isinstance<py::str>(keys)) {
                const auto key = keys.cast<std::string>();
                const auto missing = node.embedding_keys_undone({key});
                return std::vector<std::string>(missing.begin(), missing.end());
            }
            std::set<std::string> key_set;
            for (auto item : keys) {
                key_set.insert(py::cast<std::string>(item));
            }
            const auto missing = node.embedding_keys_undone(key_set);
            return std::vector<std::string>(missing.begin(), missing.end());
        })
        .def("do_embedding", [](lazyllm::DocNode& node, const py::object& embed) {
            py::dict embed_dict = py::dict(embed);
            const auto text = node.get_text(lazyllm::MetadataMode::EMBED);
            for (auto item : embed_dict) {
                const std::string key = py::cast<std::string>(item.first);
                py::object func = py::reinterpret_borrow<py::object>(item.second);
                py::object result = func(py::str(text));
                node.set_embedding_vec(key, result.cast<std::vector<double>>());
            }
        }, py::arg("embed"))
        .def("set_embedding", [](lazyllm::DocNode& node, const std::string& key, const std::vector<double>& value) {
            node.set_embedding_vec(key, value);
        })
        .def("check_embedding_state", [](lazyllm::DocNode& node, const std::string& key) {
            while (true) {
                if (node._embedding_vecs.find(key) != node._embedding_vecs.end()) {
                    node._pending_embedding_keys.erase(key);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        })
        .def("get_content", [](py::object self) {
            return self.cast<lazyllm::DocNode&>().get_text(lazyllm::MetadataMode::LLM);
        })
        .def("get_metadata_str", [](py::object self, const py::object& mode) {
            auto& node = self.cast<lazyllm::DocNode&>();
            if (mode.is_none()) return node.get_metadata_string(lazyllm::MetadataMode::ALL);
            return node.get_metadata_string(pyu::ParseMetadataMode(mode));
        }, py::arg("mode") = py::none())
        .def("get_text", [](py::object self, const py::object& metadata_mode) {
            return self.cast<lazyllm::DocNode&>().get_text(pyu::ParseMetadataMode(metadata_mode));
        }, py::arg("metadata_mode") = py::none())
        .def("to_dict", [](py::object self) {
            auto& node = self.cast<lazyllm::DocNode&>();
            py::dict d;
            d["content"] = node.get_text(lazyllm::MetadataMode::NONE);
            d["embedding"] = node._embedding_vecs;
            d["metadata"] = MetadataToPy(node._metadata);
            return d;
        })
        .def("with_score", [](const lazyllm::DocNode& node, double score) {
            lazyllm::DocNode out = node;
            out._relevance_score = score;
            return out;
        })
        .def("with_sim_score", [](const lazyllm::DocNode& node, double score) {
            lazyllm::DocNode out = node;
            out._similarity_score = score;
            return out;
        });
}

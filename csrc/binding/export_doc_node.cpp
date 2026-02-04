#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>

#include "lazyllm.hpp"
#include "document_store.hpp"
#include "doc_node.hpp"
#include "utils.hpp"

namespace py = pybind11;

lazyllm::DocNode init(
    std::optional<std::string> uid,
    std::optional<std::variant<std::string, std::vector<std::string>>> content,
    std::optional<std::string> group,
    std::optional<lazyllm::DocNode::EmbeddingVecs> embedding,
    std::optional<std::variant<std::string, py::object>> parent,
    py::object store,
    std::optional<std::unordered_map<
        std::string, std::unordered_map<std::string, std::any>>> node_groups,
    std::optional<lazyllm::DocNode::Metadata> metadata,
    std::optional<lazyllm::DocNode::Metadata> global_metadata,
    std::optional<std::string> text
) {
    if (content && text)
        throw std::invalid_argument("`text` and `content` cannot be set at the same time.");

    lazyllm::DocNode* p_parent_node = nullptr;
    std::shared_ptr<lazyllm::DocumentStore> store_adaptor = nullptr;

    // Build node groups map.
    // Usually, parent + store + node_groups are not None at the same time.
    if (parent && !store.is_none() && node_groups && global_metadata && group) {
        std::unordered_map<std::string, lazyllm::NodeGroup> node_groups_map;
        node_groups_map.reserve(node_groups->size());
        for (const auto& [group_key, group_dict] : *node_groups) {
            node_groups_map.emplace(group_key, lazyllm::NodeGroup(
                std::any_cast<std::string>(group_dict.at(std::string("parent"))),
                std::any_cast<std::string>(group_dict.at(std::string("display_name")))
            ));
        }
        store_adaptor = lazyllm::DocumentStore::from_store(store, node_groups_map);

        auto kb_id = std::any_cast<std::string>((*global_metadata).at(
            std::string(lazyllm::RAG_KEY_KB_ID)));
        auto doc_id = std::any_cast<std::string>((*global_metadata).at(
            std::string(lazyllm::RAG_KEY_DOC_ID)));

        if (const auto* parent_uid = std::get_if<std::string>(&*parent)) {
            p_parent_node = std::any_cast<lazyllm::DocNode*>(store_adaptor->call("get_node",
                {{"group_name", *group}, {"uid", *parent_uid}, {"kb_id", kb_id}}));
        }
        else
            p_parent_node = std::get<py::object>(*parent).cast<lazyllm::DocNode*>();
    }

    std::string raw_text;


    lazyllm::DocNode node(
        "",
        group.value_or(""),
        uid.value_or(""),
        p_parent_node,
        metadata.value_or(lazyllm::DocNode::Metadata()),
        std::make_shared<lazyllm::DocNode::Metadata>(
            global_metadata.value_or(lazyllm::DocNode::Metadata()))
    );
    if (store_adaptor) node.set_store(store_adaptor);
    if (embedding) {
        for (const auto& [key, vec] : *embedding)
            node.set_embedding_vec(key, vec);
    }
    if (content) {
        if (const auto* s = std::get_if<std::string>(&*content))
            node.set_root_text(std::move(*s));
        else
            node.set_root_texts(std::get<std::vector<std::string>>(*content));
    }
    else if (text){
        node.set_root_text(std::move(*text));
    }

    return node;
}

std::string DocNodeToString(const lazyllm::DocNode& node) {
    py::dict d;
    const auto children = node.py_get_children();
    for (const auto& [group, nodes] : children) {
        py::list ids;
        for (const auto* n : nodes) {
            if (n) ids.append(n->get_uid());
        }
        d[py::str(group)] = std::move(ids);
    }
    const std::string children_str = py::str(d).cast<std::string>();
    return "DocNode(id: " + node.get_uid() + ", group: " + node.get_group_name()
        + ", content: " + node.get_text(lazyllm::MetadataMode::NONE)
        + ") parent: " + node.get_parent_uid() + ", children: " + children_str;
}

void exportDocNode(py::module& m) {
    py::enum_<lazyllm::MetadataMode>(m, "MetadataMode")
        .value("ALL", lazyllm::MetadataMode::ALL)
        .value("EMBED", lazyllm::MetadataMode::EMBED)
        .value("LLM", lazyllm::MetadataMode::LLM)
        .value("NONE", lazyllm::MetadataMode::NONE);

    py::class_<lazyllm::DocNode>(m, "DocNode")
        .def(py::init(&init),
            py::kw_only(),
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
        .def_property_readonly("uid", &lazyllm::DocNode::get_uid)
        .def_property_readonly("group", &lazyllm::DocNode::get_group_name)
        .def_property("content",
            [](const lazyllm::DocNode& node) {
                return std::string(node.get_text(lazyllm::MetadataMode::NONE));
            },
            [](lazyllm::DocNode& node, const std::variant<std::string, std::vector<std::string>>& content) {
                if (const auto* content_str = std::get_if<std::string>(&content)) {
                    node.set_root_text(std::move(*content_str));
                    return;
                }
                else {
                    node.set_root_texts(std::get<std::vector<std::string>>(content));
                    return;
                }
            }
        )
        .def_property("number",
            [](const lazyllm::DocNode& node) {
                const auto it = node._metadata.find("lazyllm_store_num");
                if (it == node._metadata.end()) return 0;
                return std::any_cast<int>(it->second);
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
        .def_property("parent",
            [](const lazyllm::DocNode& node) { return node.get_parent_node(); },
            [](lazyllm::DocNode& node, lazyllm::DocNode* parent) { node.set_parent_node(parent); },
            py::return_value_policy::reference
        )
        .def_property("children",
            [](const lazyllm::DocNode& node) { return node.py_get_children(); },
            [](lazyllm::DocNode& node, const lazyllm::DocNode::Children& children) {
                node.set_children(children);
            }
        )
        .def_property_readonly("root_node",
            [](const lazyllm::DocNode& node) { return node.get_root_node(); },
            py::return_value_policy::reference
        )
        .def_property_readonly("is_root_node",
            [](const lazyllm::DocNode& node) { return node.get_parent_node() == nullptr; }
        )
        .def_property("global_metadata",
            [](const lazyllm::DocNode& node) { return *(node.get_root_node()->_p_global_metadata); },
            [](lazyllm::DocNode& node, const lazyllm::DocNode::Metadata& meta) {
                node._p_global_metadata = std::make_shared<lazyllm::DocNode::Metadata>(meta);
            }
        )
        .def_property("metadata",
            [](const lazyllm::DocNode& node) { return node._metadata; },
            [](lazyllm::DocNode& node, const lazyllm::DocNode::Metadata& meta) { node._metadata = meta; }
        )
        .def_property("excluded_embed_metadata_keys",
            [](const lazyllm::DocNode& node) { return node.get_excluded_embed_metadata_keys(); },
            [](lazyllm::DocNode& node, const std::set<std::string>& keys) {
                node.set_excluded_embed_metadata_keys(keys);
            }
        )
        .def_property("excluded_llm_metadata_keys",
            [](const lazyllm::DocNode& node) { return node.get_excluded_llm_metadata_keys(); },
            [](lazyllm::DocNode& node, const std::set<std::string>& keys) {
                node.set_excluded_llm_metadata_keys(keys);
            }
        )
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
            const auto children = node.py_get_children();
            for (const auto& [group, nodes] : children) {
                py::list ids;
                for (const auto* n : nodes) if (n) ids.append(n->get_uid());
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
            st["_group"] = node.get_group_name();
            st["_embedding"] = node._embedding_vecs;
            st["_metadata"] = node._metadata;
            st["_global_metadata"] = *(node._p_global_metadata);
            st["_excluded_embed_metadata_keys"] = node.get_excluded_embed_metadata_keys();
            st["_excluded_llm_metadata_keys"] = node.get_excluded_llm_metadata_keys();
            st["_store"] = py::none();
            st["_node_groups"] = py::none();
            return st;
        })
        .def("has_missing_embedding", [](const lazyllm::DocNode& node,
            std::variant<std::string, std::vector<std::string>>& keys) {
            if (const auto& single_key = std::get_if<std::string>(&keys))
                return node.embedding_keys_undone({*single_key});
            else {
                const auto& key_list = std::get<std::vector<std::string>>(keys);
                return node.embedding_keys_undone(std::set<std::string>(key_list.begin(), key_list.end()));
            }
        })
        .def("do_embedding", &lazyllm::DocNode::py_do_embedding, py::arg("embed"))
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
        .def("get_content", [](const lazyllm::DocNode& node) { return node.get_text(lazyllm::MetadataMode::LLM); })
        .def("get_metadata_str", &lazyllm::DocNode::get_metadata_string,
            py::arg("mode") = lazyllm::MetadataMode::ALL)
        .def("get_text", &lazyllm::DocNode::get_text, py::arg("metadata_mode") = lazyllm::MetadataMode::NONE)
        .def("to_dict", [](const lazyllm::DocNode& node) {
            py::dict d;
            d["content"] = node.get_text(lazyllm::MetadataMode::NONE);
            d["embedding"] = node._embedding_vecs;
            d["metadata"] = node._metadata;
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

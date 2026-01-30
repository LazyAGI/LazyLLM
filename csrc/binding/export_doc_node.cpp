#include <algorithm>
#include <cctype>

#include "lazyllm.hpp"
#include "document_store.hpp"
#include "doc_node.hpp"
#include "utils.hpp"

namespace py = pybind11;

lazyllm::DocNode init(
    std::optional<std::string> uid,
    std::optional<std::variant<std::string, std::vector<std::string>>> content,
    std::optional<std::string> group,
    std::optional<lazyllm::DocNode::EmbeddingVec> embedding,
    std::optional<std::variant<std::string, py::object>> parent,
    py::object store,
    std::optional<std::unordered_map<
        std::string, std::unordered_map<std::any, std::any>>> node_groups,
    std::optional<lazyllm::DocNode::Metadata> metadata,
    std::optional<lazyllm::DocNode::Metadata> global_metadata,
    std::optional<std::string> text
) {
    if (content && text) {
        throw std::invalid_argument("`text` and `content` cannot be set at the same time.");
    }

    // Find parent node.
    lazyllm::DocNode* p_parent_node = nullptr;
    std::shared_ptr<lazyllm::DocumentStore> store_bridge = nullptr;

    // Build node groups map.
    // Usually, parent + store + node_groups are not None at the same time.
    if (parent && !store.is_none() && node_groups && global_metadata && group) {
        std::unordered_map<std::string, lazyllm::NodeGroup> node_groups_map;
        node_groups_map.reserve(node_groups->size());
        for (const auto &entry : *node_groups) {
            auto &group_key = entry.first;
            auto &group_dict = entry.second;
            node_groups_map.emplace(group_key, lazyllm::NodeGroup(
                std::any_cast<std::string>(group_dict.at(std::string("parent"))),
                std::any_cast<std::string>(group_dict.at(std::string("display_name")))
            ));
        }
        auto store_bridge = lazyllm::DocumentStore::from_store(store, node_groups_map);

        auto kb_id = std::any_cast<std::string>((*global_metadata)[
            std::string(lazyllm::DocumentStore::RAG_KB_ID_KEY)]);
        auto doc_id = std::any_cast<std::string>((*global_metadata)[
            std::string(lazyllm::DocumentStore::RAG_DOC_ID_KEY)]);

        if (std::holds_alternative<std::string>(*parent)) {
            const std::string& parent_uid = std::get<std::string>(*parent);
            p_parent_node = store_bridge->get_node(parent_uid, *group, kb_id).cast<lazyllm::DocNode*>();
        }
        else
            p_parent_node = std::get<py::object>(*parent).cast<lazyllm::DocNode*>();
    }

    lazyllm::DocNode node(
        content.value_or(text.value_or(std::string(""))),
        uid.value_or(lazyllm::GenerateUUID()),
        group.value_or(""),
        p_parent_node,
        store_bridge,
        embedding.value_or(lazyllm::DocNode::EmbeddingVec()),
        metadata.value_or(lazyllm::DocNode::Metadata()),
        global_metadata.value_or(lazyllm::DocNode::Metadata())
    );


    if (embedding.has_value()) {
        node.set_embedding(*embedding);
    }

    if (!metadata.is_none()) {
        if (!py::isinstance<py::dict>(metadata)) {
            throw std::invalid_argument("`metadata` must be a dict.");
        }
        node.set_metadata(dict_to_metadata(metadata.cast<py::dict>()));
    }

    if (!global_metadata.is_none()) {
        if (!py::isinstance<py::dict>(global_metadata)) {
            throw std::invalid_argument("`global_metadata` must be a dict.");
        }
        node.set_global_metadata(dict_to_metadata(global_metadata.cast<py::dict>()));
    }

    if (!parent.is_none()) {
        if (py::isinstance<lazyllm::DocNode>(parent)) {
            node.set_parent(parent.cast<std::shared_ptr<lazyllm::DocNode>>());
        } else if (py::isinstance<py::str>(parent)) {
            throw std::invalid_argument("`parent` as str is not supported in C++ binding.");
        } else {
            throw std::invalid_argument("`parent` must be a DocNode or None.");
        }
    }

    std::shared_ptr<lazyllm::DocumentStore> store_bridge;
    if (!store.is_none()) {
        py::object store_obj = store;
        store_bridge = lazyllm::DocumentStore::from_store(store_obj, node_groups_map);
    }
    node.set_store(std::move(store_bridge));

    (void)node_groups;

    return node;
}

void exportDocNode(py::module& m) {
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
            py::arg("text") = py::none(),

            py::keep_alive<1, 7>() // Keep store alive
        )
        .def_property_readonly("uid", &lazyllm::DocNode::uid)
        number
        .def("set_text", &lazyllm::DocNode::set_text, py::arg("text"))
        .def("get_text", &lazyllm::DocNode::get_text);
}

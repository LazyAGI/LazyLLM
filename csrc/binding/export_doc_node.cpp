#include <algorithm>

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
    if (embedding) node.set_embedding_vec(*embedding);
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
    @property
    def number(self) -> int:
        return self._metadata.get('lazyllm_store_num', 0)
        .def("set_text", &lazyllm::DocNode::set_text, py::arg("text"))
        .def("get_text", &lazyllm::DocNode::get_text);
}

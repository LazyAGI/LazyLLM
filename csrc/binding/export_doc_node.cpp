#include "lazyllm.hpp"
#include "document_store.hpp"
#include "doc_node.hpp"

namespace py = pybind11;

lazyllm::DocNode init(
    std::optional<std::string> uid,
    std::optional<std::variant<std::string, std::vector<std::any>>> content,
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
    lazyllm::DocNode node;

    if (content && text) {
        throw std::invalid_argument("`text` and `content` cannot be set at the same time.");
    }

    if (uid) node.set_uid(uid.value());
    else node.generateUUID();

    if (content) {
        if (auto p = std::get_if<std::string>(&*content))
        else node.set_text(std::any_cast<std::string>(content.value));
    }
    else node.set_text(*text);

    if (uid.has_value() && !uid->empty()) {
        node.set_uid(*uid);
    } else {
        node.generateUUID();
    }

    if (group.has_value()) {
        node.set_group(*group);
    }

    if (!content.is_none()) {
        if (py::isinstance<py::str>(content)) {
            node.set_content(content.cast<std::string>());
        } else if (py::isinstance<py::list>(content) || py::isinstance<py::tuple>(content)) {
            node.set_content(content.cast<std::vector<std::string>>());
        } else {
            throw std::invalid_argument("`content` must be a str or list[str].");
        }
    } else if (text.has_value()) {
        node.set_text(*text);
    } else {
        node.set_text("");
    }

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
        store_bridge = std::make_shared<lazyllm::DocumentStore>(store_obj);
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
            py::arg("text") = py::none()
        )
        .def_property_readonly("uid", &lazyllm::DocNode::uid)
        .def("set_text", &lazyllm::DocNode::set_text, py::arg("text"))
        .def("get_text", &lazyllm::DocNode::get_text);
}

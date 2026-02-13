#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "adaptor_base_wrapper.hpp"
#include "doc_node.hpp"

namespace lazyllm {

struct NodeGroup {
    enum class Type {
        ORIGINAL, CHUNK, SUMMARY, IMAGE_INFO, QUESTION_ANSWER, OTHER
    };
    std::string _parent;
    std::string _display_name;
    Type _type;
    NodeGroup(
        const std::string& parent,
        const std::string& display_name,
        const Type& type = Type::ORIGINAL) :
            _parent(parent), _display_name(display_name), _type(type) {}
};

class LAZYLLM_HIDDEN DocumentStore : public AdaptorBaseWrapper {
public:
    DocumentStore() = delete;
    explicit DocumentStore(
        const pybind11::object& store,
        const std::unordered_map<std::string, NodeGroup> &map) :
            AdaptorBaseWrapper(store), _node_groups_map(map) {}

    // Cache-aware factory to avoid rebuilding adaptor for the same Python store.
    static std::shared_ptr<DocumentStore> from_store(
        const pybind11::object& store, const std::unordered_map<std::string, NodeGroup>& map) {
        if (store.is_none()) return nullptr;

        pybind11::gil_scoped_acquire gil;
        PyObject *key = store.ptr();
        auto &cache = store_cache();
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (auto existing = it->second.lock())
                return existing;
        }
        auto created = std::make_shared<DocumentStore>(store, map);
        cache[key] = created;
        return created;
    }

    DocNode::Children get_node_children(const DocNode* node) const {
        DocNode::Children out;
        auto& kb_id = std::any_cast<std::string&>(node->_p_global_metadata->at(std::string(RAGMetadataKeys::KB_ID)));
        auto& doc_id = std::any_cast<std::string&>(node->_p_global_metadata->at(std::string(RAGMetadataKeys::DOC_ID)));
        auto& group_name = node->_group_name;
        for(auto& [current_group_name, group] : _node_groups_map) {
            if (group._parent != group_name) continue;
            if (!std::any_cast<bool>(call("is_group_active", {{"group", current_group_name}}))) continue;
            auto nodes_in_group = std::any_cast<std::vector<DocNode*>>(call("get_nodes", {
                {"group_name", current_group_name},
                {"kb_id", kb_id},
                {"doc_ids", std::vector<std::string>({doc_id})}
            }));

            std::vector<DocNode*> children;
            children.reserve(nodes_in_group.size());
            for (auto* n : nodes_in_group)
                if (n->get_parent_node() == node) children.push_back(n);
            out[current_group_name] = children;
        }
        return out;
    }

private:
    std::unordered_map<std::string, NodeGroup> _node_groups_map;

    std::any call_impl(
        const std::string& func_name,
        const pybind11::object& func,
        const std::unordered_map<std::string, std::any>& args) const override
    {
        if (func_name == "is_group_active") {
            return func(args.at("group")).cast<bool>();
        }
        else if (func_name == "get_node") {
            return func(
                pybind11::arg("group_name") = std::any_cast<std::string>(args.at("group_name")),
                pybind11::arg("uids") = std::vector<std::string>({std::any_cast<std::string>(args.at("uid"))}),
                pybind11::arg("kb_id") = std::any_cast<std::string>(args.at("kb_id")),
                pybind11::arg("display") = true
            ).cast<pybind11::list>()[0].cast<DocNode*>();
        }
        else if (func_name == "get_nodes") {
            return func(
                pybind11::arg("group_name") = std::any_cast<std::string>(args.at("group_name")),
                pybind11::arg("kb_id") = std::any_cast<std::string>(args.at("kb_id")),
                pybind11::arg("doc_ids") = std::vector<std::string>({std::any_cast<std::string>(args.at("doc_id"))})
            ).cast<std::vector<DocNode*>>();
        }
        else if (func_name == "get_node_children") {
            return get_node_children(std::any_cast<DocNode*>(args.at("node")));
        }

        throw std::runtime_error("Unknown DocumentStore function: " + func_name);
    }

    // Cache by Python object identity to ensure one wrapper per store instance.
    static std::unordered_map<PyObject *, std::weak_ptr<DocumentStore>> &store_cache() {
        static std::unordered_map<PyObject *, std::weak_ptr<DocumentStore>> cache;
        return cache;
    }
};

} // namespace lazyllm

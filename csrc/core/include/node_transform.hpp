#pragma once

#include <any>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "doc_node.hpp"

namespace lazyllm {

class NodeTransform {
public:
    using TransformKwargs = std::unordered_map<std::string, std::any>;
    using TransformItem = std::variant<std::string, DocNode*>;
    using TransformResult = std::vector<TransformItem>;

    explicit NodeTransform(int worker_num = 0) : _worker_num(worker_num) {}
    virtual ~NodeTransform() = default;

    std::vector<DocNode*> batch_forward(
        const std::vector<DocNode*>& documents,
        const std::string& node_group,
        const TransformKwargs& kwargs = {})
    {
        std::vector<DocNode*> results;
        for (auto* node : documents) {
            if (node == nullptr) continue;
            auto children = node->py_get_children();
            if (children.find(node_group) != children.end()) continue;

            auto splits = forward(node, node_group, kwargs);
            children[node_group] = splits;
            node->set_children(children);
            results.insert(results.end(), splits.begin(), splits.end());
        }
        return results;
    }

    std::vector<DocNode*> batch_forward(
        DocNode* document,
        const std::string& node_group,
        const TransformKwargs& kwargs = {})
    {
        if (document == nullptr) return {};
        return batch_forward(std::vector<DocNode*>{document}, node_group, kwargs);
    }

    virtual TransformResult transform(DocNode* document, const TransformKwargs& kwargs) = 0;

    NodeTransform& with_name(const std::optional<std::string>& name, bool /*copy*/ = true) {
        if (name.has_value()) _name = *name;
        return *this;
    }

    const std::string& name() const { return _name; }
    int worker_num() const { return _worker_num; }

protected:
    std::vector<DocNode*> forward(
        DocNode* node,
        const std::string& node_group,
        const TransformKwargs& kwargs)
    {
        TransformResult raw = transform(node, kwargs);
        std::vector<DocNode*> out;
        out.reserve(raw.size());

        for (auto& item : raw) {
            if (auto* text = std::get_if<std::string>(&item)) {
                if (text->empty()) continue;
                auto child = std::make_unique<DocNode>("", node_group, "", node);
                child->set_root_text(std::move(*text));
                auto* ptr = child.get();
                _owned_nodes.emplace_back(std::move(child));
                out.push_back(ptr);
            } else {
                auto* child = std::get<DocNode*>(item);
                if (child == nullptr) continue;
                child->set_parent_node(node);
                out.push_back(child);
            }
        }
        return out;
    }

protected:
    int _worker_num = 0;
    std::string _name;
    std::vector<std::unique_ptr<DocNode>> _owned_nodes;
};

} // namespace lazyllm

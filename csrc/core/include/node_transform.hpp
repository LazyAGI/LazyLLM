#pragma once

#include <any>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "doc_node.hpp"
#include "thread_pool.hpp"

namespace lazyllm {

class NodeTransform {
public:
    std::string _name = "";

    explicit NodeTransform(int worker_num = 0) : _worker_num(worker_num) {}
    virtual ~NodeTransform() = default;

    virtual std::vector<DocNode> transform(const DocNode*) const = 0;
    std::vector<DocNode> operator()(const DocNode& node) const { return transform(&node); }

    std::vector<DocNode*> batch_forward(std::vector<DocNode*>& nodes, const std::string& node_group_name) {
        std::vector<DocNode*> whole_nodes;
        if (nodes.empty()) return whole_nodes;

        if (_worker_num > 0) {
            ThreadPool pool(static_cast<size_t>(_worker_num));
            std::vector<std::future<std::vector<DocNode*>>> futures;
            futures.reserve(nodes.size());
            for (auto* p_node : nodes) {
                futures.emplace_back(pool.enqueue(
                    [this, p_node, node_group_name] { return forward(p_node, node_group_name); }));
            }

            for (auto& fut : futures) {
                auto parts = fut.get();
                whole_nodes.insert(whole_nodes.end(), parts.begin(), parts.end());
            }
        } else {
            for (auto* p_node : nodes) {
                auto parts = forward(p_node, node_group_name);
                whole_nodes.insert(whole_nodes.end(), parts.begin(), parts.end());
            }
        }

        return whole_nodes;
    }

    int worker_num() const { return _worker_num; }

private:
    std::vector<DocNode*> forward(DocNode* p_node, const std::string& node_group_name) {
        if (p_node->is_children_group_exists(node_group_name)) return {};

        auto raw_nodes = transform(p_node);
        std::vector<DocNode*> out;
        out.reserve(raw_nodes.size());

        for (auto& node_ : raw_nodes) {
            node_.set_parent_node(p_node);
            node_._group_name = node_group_name;
            auto child = std::make_unique<DocNode>(std::move(node_));
            auto* ptr = child.get();
            {
                std::lock_guard<std::mutex> lock(_owned_nodes_mutex);
                _owned_nodes.emplace_back(std::move(child));
            }
            out.push_back(ptr);
        }
        p_node->set_children_group(node_group_name, out);

        return out;
    }

    std::vector<std::unique_ptr<DocNode>> _owned_nodes;
    std::mutex _owned_nodes_mutex;
    int _worker_num = 0;
    bool _support_rich = false;
};

} // namespace lazyllm

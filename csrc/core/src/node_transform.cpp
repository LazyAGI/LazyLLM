#include "node_transform.hpp"

namespace lazyllm {

std::vector<PDocNode> NodeTransform::batch_forward(
    std::vector<PDocNode>& nodes, const std::string& node_group_name
) {
    std::vector<PDocNode> whole_nodes;
    if (nodes.empty()) return whole_nodes;

    if (_worker_num > 0) {
        ThreadPool pool(static_cast<size_t>(_worker_num));
        std::vector<std::future<std::vector<PDocNode>>> futures;
        futures.reserve(nodes.size());
        for (auto node : nodes) {
            futures.emplace_back(pool.enqueue(
                [this, node, node_group_name]() { return forward(node, node_group_name); }));
        }

        for (auto& fut : futures) {
            auto parts = fut.get();
            whole_nodes.insert(whole_nodes.end(), parts.begin(), parts.end());
        }
    } else {
        for (auto node : nodes) {
            auto parts = forward(node, node_group_name);
            whole_nodes.insert(whole_nodes.end(), parts.begin(), parts.end());
        }
    }

    return whole_nodes;
}

std::vector<PDocNode> NodeTransform::forward(PDocNode node, const std::string& node_group_name) {
    auto p_nodes = transform(node);
    for (auto& p_node : p_nodes) {
        p_node->set_parent_node(&*node);
        p_node->_group_name = node_group_name;
    }
    node->set_children_group(node_group_name, p_nodes);

    return p_nodes;
}

} // namespace lazyllm

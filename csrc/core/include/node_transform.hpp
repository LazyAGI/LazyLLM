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

    virtual std::vector<PDocNode> transform(const PDocNode) const = 0;
    std::vector<PDocNode> operator()(const PDocNode node) const { return transform(node); }
    std::vector<PDocNode> batch_forward(std::vector<PDocNode>& nodes, const std::string& node_group_name);

    int worker_num() const { return _worker_num; }

private:
    std::vector<PDocNode> forward(PDocNode node, const std::string& node_group_name);

    std::mutex _lock;
    int _worker_num = 0;
};

} // namespace lazyllm

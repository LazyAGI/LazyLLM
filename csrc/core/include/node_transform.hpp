#pragma once

#include <memory>
#include <string>
#include <vector>

#include "doc_node.hpp"

namespace lazyllm {

class NodeTransform {
public:
    std::string _name = "";
    virtual ~NodeTransform() = default;

    virtual std::vector<PDocNode> transform(const PDocNode) const = 0;
};

} // namespace lazyllm

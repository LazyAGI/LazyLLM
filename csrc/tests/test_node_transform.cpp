#include <gtest/gtest.h>

#include "doc_node.hpp"
#include "node_transform.hpp"

using namespace lazyllm;

namespace {

class PairTransform final : public NodeTransform {
public:
    PairTransform() = default;

    mutable int transform_calls = 0;

    std::vector<PDocNode> transform(PDocNode node) const override {
        ++transform_calls;
        const auto& shared_meta = node->get_root_node()->_p_global_metadata;
        return {
            std::make_shared<DocNode>(std::string(node->get_text_view()) + "_A", shared_meta),
            std::make_shared<DocNode>(std::string(node->get_text_view()) + "_B", shared_meta)
        };
    }
};

std::vector<PDocNode> make_roots(size_t count) {
    std::vector<PDocNode> roots;
    roots.reserve(count);
    for (size_t i = 0; i < count; ++i)
        roots.push_back(std::make_shared<DocNode>("n" + std::to_string(i), "", "uid_" + std::to_string(i)));
    return roots;
}

} // namespace

TEST(node_transform, transform_returns_result) {
    auto roots = make_roots(1);
    PairTransform transform;
    const auto direct = transform.transform(roots[0]);
    EXPECT_EQ(direct.size(), 2u);
}

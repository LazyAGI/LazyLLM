#include <gtest/gtest.h>

#include "doc_node.hpp"
#include "node_transform.hpp"

using namespace lazyllm;

namespace {

class PairTransform final : public NodeTransform {
public:
    explicit PairTransform(int worker_num = 0) : NodeTransform(worker_num) {}

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

class SingleTransform final : public NodeTransform {
public:
    explicit SingleTransform(int worker_num) : NodeTransform(worker_num) {}

    std::vector<PDocNode> transform(PDocNode node) const override {
        const auto shared_meta = node->get_root_node()->_p_global_metadata;
        return {
            std::make_shared<DocNode>(std::string(node->get_text_view()) + "_child", shared_meta)
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

TEST(node_transform, call_operator_returns_transform_result) {
    auto roots = make_roots(1);
    PairTransform transform;
    const auto direct = transform(roots[0]);
    EXPECT_EQ(direct.size(), 2u);
}

TEST(node_transform, batch_forward) {
    auto roots = make_roots(2);
    PairTransform transform;
    const auto children = transform.batch_forward(roots, "split");

    EXPECT_EQ(children.size(), 4u);
    for (const auto* child : children) {
        EXPECT_NE(child->get_parent_node(), nullptr);
        EXPECT_EQ(child->_group_name, "split");
    }

    for (const auto& root : roots) {
        EXPECT_TRUE(root.is_children_group_exists("split"));
        EXPECT_EQ(root.get_children().at("split").size(), 2u);
    }
}

TEST(node_transform, batch_forward_parallel_mode_respects_worker_num) {
    auto roots = make_roots(8);
    SingleTransform transform(3);
    EXPECT_EQ(transform.worker_num(), 3);

    const auto children = transform.batch_forward(roots, "parallel");
    EXPECT_EQ(children.size(), roots.size());
    for (const auto& root : roots) {
        EXPECT_TRUE(root.is_children_group_exists("parallel"));
        EXPECT_EQ(root.get_children().at("parallel").size(), 1u);
    }
}

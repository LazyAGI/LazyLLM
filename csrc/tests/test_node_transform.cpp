#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "doc_node.hpp"
#include "node_transform.hpp"

namespace {

class PairTransform final : public lazyllm::NodeTransform {
public:
    explicit PairTransform(int worker_num = 0) : lazyllm::NodeTransform(worker_num) {}

    mutable int transform_calls = 0;

    std::vector<lazyllm::DocNode> transform(const lazyllm::DocNode* node) const override {
        ++transform_calls;
        const auto shared_meta = node->get_root_node()->_p_global_metadata;
        return {
            lazyllm::DocNode(
                std::string(node->get_text_view()) + "_A",
                "",
                "",
                nullptr,
                {},
                shared_meta),
            lazyllm::DocNode(
                std::string(node->get_text_view()) + "_B",
                "",
                "",
                nullptr,
                {},
                shared_meta),
        };
    }
};

class SingleTransform final : public lazyllm::NodeTransform {
public:
    explicit SingleTransform(int worker_num) : lazyllm::NodeTransform(worker_num) {}

    std::vector<lazyllm::DocNode> transform(const lazyllm::DocNode* node) const override {
        const auto shared_meta = node->get_root_node()->_p_global_metadata;
        return {
            lazyllm::DocNode(
                std::string(node->get_text_view()) + "_child",
                "",
                "",
                nullptr,
                {},
                shared_meta)
        };
    }
};

} // namespace

TEST(NodeTransform, BatchForwardCreatesChildrenAndSetsParentGroup) {
    auto shared_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    std::vector<lazyllm::DocNode> roots;
    roots.reserve(2);
    roots.emplace_back("left", "", "left_uid", nullptr, lazyllm::DocNode::Metadata(), shared_meta);
    roots.emplace_back("right", "", "right_uid", nullptr, lazyllm::DocNode::Metadata(), shared_meta);

    std::vector<lazyllm::DocNode*> root_ptrs{&roots[0], &roots[1]};
    PairTransform transform(0);
    const auto direct = transform(roots[0]);
    EXPECT_EQ(direct.size(), 2u);

    const auto children = transform.batch_forward(root_ptrs, "split");
    ASSERT_EQ(children.size(), 4u);

    for (const auto* child : children) {
        EXPECT_NE(child->get_parent_node(), nullptr);
        EXPECT_EQ(child->_group_name, "split");
    }
    for (const auto& root : roots) {
        EXPECT_TRUE(root.is_children_group_exists("split"));
        EXPECT_EQ(root.py_get_children().at("split").size(), 2u);
    }
}

TEST(NodeTransform, BatchForwardSkipsExistingGroup) {
    auto shared_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode root("root", "", "uid", nullptr, lazyllm::DocNode::Metadata(), shared_meta);
    root.set_children_group("already", {});
    std::vector<lazyllm::DocNode*> root_ptrs{&root};

    PairTransform transform(0);
    const auto children = transform.batch_forward(root_ptrs, "already");

    EXPECT_TRUE(children.empty());
    EXPECT_EQ(transform.transform_calls, 0);
}

TEST(NodeTransform, BatchForwardSupportsParallelMode) {
    auto shared_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    std::vector<lazyllm::DocNode> roots;
    roots.reserve(8);
    for (int i = 0; i < 8; ++i) {
        roots.emplace_back(
            "n" + std::to_string(i),
            "",
            "uid_" + std::to_string(i),
            nullptr,
            lazyllm::DocNode::Metadata(),
            shared_meta);
    }

    std::vector<lazyllm::DocNode*> root_ptrs;
    root_ptrs.reserve(roots.size());
    for (auto& root : roots) root_ptrs.push_back(&root);

    SingleTransform transform(3);
    const auto children = transform.batch_forward(root_ptrs, "parallel");

    EXPECT_EQ(transform.worker_num(), 3);
    EXPECT_EQ(children.size(), roots.size());
    for (const auto& root : roots) {
        EXPECT_TRUE(root.is_children_group_exists("parallel"));
        EXPECT_EQ(root.py_get_children().at("parallel").size(), 1u);
    }
}

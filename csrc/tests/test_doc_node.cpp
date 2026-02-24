#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "adaptor_base.hpp"
#include "doc_node.hpp"
#include "utils.hpp"

namespace {

class CountingAdaptor final : public lazyllm::AdaptorBase {
public:
    mutable int call_count = 0;
    lazyllm::DocNode::Children to_return;

    std::any call(
        const std::string& func_name,
        const std::unordered_map<std::string, std::any>& args) const override
    {
        ++call_count;
        EXPECT_EQ(func_name, "get_node_children");
        EXPECT_TRUE(args.find("node") != args.end());
        return to_return;
    }
};

} // namespace

TEST(DocNode, ConstructorAndTextHashUpdate) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode node("hello", "group", "fixed-uid", nullptr, {}, global_meta);

    EXPECT_EQ(node.get_uid(), "fixed-uid");
    EXPECT_EQ(node._group_name, "group");
    EXPECT_EQ(node.get_text(), "hello");

    const size_t old_hash = node.get_text_hash();
    node.set_text_view("world");
    EXPECT_NE(node.get_text_hash(), old_hash);
    EXPECT_EQ(node.get_text(), "world");
}

TEST(DocNode, MetadataModesAndTextRender) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode::Metadata metadata{
        {"alpha", std::string("A")},
        {"beta", std::string("B")},
    };
    lazyllm::DocNode node("body", "", "", nullptr, metadata, global_meta);

    node.set_excluded_embed_metadata_keys({"beta"});
    node.set_excluded_llm_metadata_keys({"alpha"});

    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::ALL), "alpha:A\nbeta:B");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::EMBED), "alpha:A");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::LLM), "beta:B");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::NONE), "");

    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "body");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::EMBED), "alpha:A\n\nbody");
}

TEST(DocNode, ParentChildrenAndUidViews) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode root("root", "root_group", "root_uid", nullptr, {}, global_meta);
    lazyllm::DocNode child("child", "child_group", "child_uid", &root, {}, global_meta);

    root.set_children_group("split", {&child});

    EXPECT_EQ(child.get_root_node(), &root);
    EXPECT_EQ(child.get_parent_uid(), "root_uid");
    EXPECT_TRUE(root.is_children_group_exists("split"));

    const auto child_ids = root.py_get_children_uid();
    ASSERT_TRUE(child_ids.find("split") != child_ids.end());
    ASSERT_EQ(child_ids.at("split").size(), 1u);
    EXPECT_EQ(child_ids.at("split")[0], "child_uid");
}

TEST(DocNode, StoreBackedChildrenAreCached) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode parent("p", "", "parent", nullptr, {}, global_meta);
    lazyllm::DocNode child("c", "", "child", &parent, {}, global_meta);

    auto adaptor = std::make_shared<CountingAdaptor>();
    adaptor->to_return["cached"] = {&child};
    parent.set_store(adaptor);

    const auto first = parent.py_get_children();
    const auto second = parent.py_get_children();

    EXPECT_EQ(adaptor->call_count, 1);
    ASSERT_TRUE(first.find("cached") != first.end());
    ASSERT_EQ(first.at("cached").size(), 1u);
    EXPECT_EQ(first.at("cached")[0], &child);
    EXPECT_EQ(second.at("cached")[0], &child);
}

TEST(DocNode, GlobalDocPathAndExclusionInheritance) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    (*global_meta)[std::string(lazyllm::RAGMetadataKeys::DOC_PATH)] = std::string("/tmp/a.txt");

    lazyllm::DocNode root("root", "", "root", nullptr, {}, global_meta);
    lazyllm::DocNode child("child", "", "child", &root, {}, global_meta);

    root.set_excluded_embed_metadata_keys({"root_embed"});
    child.set_excluded_embed_metadata_keys({"child_embed"});
    root.set_excluded_llm_metadata_keys({"root_llm"});
    child.set_excluded_llm_metadata_keys({"child_llm"});

    EXPECT_EQ(child.get_doc_path(), "/tmp/a.txt");
    child.set_doc_path("/tmp/b.txt");
    EXPECT_EQ(root.get_doc_path(), "/tmp/b.txt");

    EXPECT_EQ(
        child.get_excluded_embed_metadata_keys(),
        (std::set<std::string>{"child_embed", "root_embed"}));
    EXPECT_EQ(
        child.get_excluded_llm_metadata_keys(),
        (std::set<std::string>{"child_llm", "root_llm"}));
}

TEST(DocNode, EmbeddingHelpers) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode node("text", "", "node", nullptr, {}, global_meta);

    EXPECT_THROW(node.embedding_keys_undone({}), std::runtime_error);

    node.set_embedding_vec("done", {1.0, 2.0});
    const auto missing = node.embedding_keys_undone({"done", "todo"});
    EXPECT_EQ(missing, (std::set<std::string>{"todo"}));

    node.py_do_embedding({
        {"len_embedding", [](const std::string& input) {
            return std::vector<double>{static_cast<double>(input.size())};
        }}
    });
    ASSERT_TRUE(node._embedding_vecs.find("len_embedding") != node._embedding_vecs.end());
    ASSERT_EQ(node._embedding_vecs["len_embedding"].size(), 1u);
    EXPECT_EQ(node._embedding_vecs["len_embedding"][0], 6.0);
}

TEST(DocNode, EqualityUsesUid) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode lhs("left", "", "same_uid", nullptr, {}, global_meta);
    lazyllm::DocNode rhs("right", "", "same_uid", nullptr, {}, global_meta);
    lazyllm::DocNode other("other", "", "other_uid", nullptr, {}, global_meta);

    EXPECT_TRUE(lhs == rhs);
    EXPECT_FALSE(lhs != rhs);
    EXPECT_TRUE(lhs != other);
}

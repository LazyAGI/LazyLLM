#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "adaptor_base.hpp"
#include "doc_node.hpp"
#include "utils.hpp"

namespace {

class MockChildrenAdaptor final : public lazyllm::AdaptorBase {
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

TEST(doc_node, constructor) {
    lazyllm::DocNode node("hello", "group", "fixed-uid");
    EXPECT_EQ(node.get_uid(), "fixed-uid");
    EXPECT_EQ(node._group_name, "group");
    EXPECT_EQ(node.get_text(), "hello");
}

TEST(doc_node, set_text_view_updates_text_hash) {
    lazyllm::DocNode node("hello", "group", "fixed-uid");
    const size_t old_hash = node.get_text_hash();
    node.set_text_view("world");
    EXPECT_NE(node.get_text_hash(), old_hash);
    EXPECT_EQ(node.get_text(), "world");
}

TEST(doc_node, metadata) {
    lazyllm::DocNode node;
    node._metadata = lazyllm::DocNode::Metadata{
        {"alpha", std::string("A")},
        {"beta", std::string("B")},
    };
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::ALL), "alpha: A\nbeta: B");

    node.set_excluded_embed_metadata_keys({"beta"});
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::EMBED), "alpha: A");

    node.set_excluded_llm_metadata_keys({"alpha"});
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::LLM), "beta: B");

    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::NONE), "");

    lazyllm::DocNode root("root");
    lazyllm::DocNode child("child", "", "", &root);

    root.set_excluded_embed_metadata_keys({"root_embed"});
    child.set_excluded_embed_metadata_keys({"child_embed"});
    EXPECT_EQ(
        child.get_excluded_embed_metadata_keys(),
        (std::set<std::string>{"child_embed", "root_embed"}));

    root.set_excluded_llm_metadata_keys({"root_llm"});
    child.set_excluded_llm_metadata_keys({"child_llm"});
    EXPECT_EQ(
        child.get_excluded_llm_metadata_keys(),
        (std::set<std::string>{"child_llm", "root_llm"}));
}

TEST(doc_node, text) {
    lazyllm::DocNode node("body");
    node._metadata = lazyllm::DocNode::Metadata{{"alpha", std::string("A")}};
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "body");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::EMBED), "alpha: A\n\nbody");
}

TEST(doc_node, relationships) {
    lazyllm::DocNode root("root", "root_group", "root_uid");
    lazyllm::DocNode child("child", "child_group", "child_uid", &root);
    EXPECT_EQ(child.get_root_node(), &root);
    EXPECT_EQ(child.get_parent_uid(), "root_uid");

    root.set_children_group("split", {std::make_shared<lazyllm::DocNode>(std::move(child))});
    EXPECT_TRUE(root.is_children_group_exists("split"));
    const auto child_ids = root.get_children_uid();
    ASSERT_TRUE(child_ids.find("split") != child_ids.end());
    ASSERT_EQ(child_ids.at("split").size(), 1u);
    EXPECT_EQ(child_ids.at("split")[0], "child_uid");
}

TEST(doc_node, children_caching) {
    lazyllm::DocNode parent("p", "", "parent");
    lazyllm::DocNode child("c", "", "child", &parent);

    auto adaptor = std::make_shared<MockChildrenAdaptor>();
    adaptor->to_return["cached"] = {std::make_shared<lazyllm::DocNode>(std::move(child))};
    parent.set_store(adaptor);

    const auto first = parent.get_children();
    const auto second = parent.get_children();

    EXPECT_EQ(adaptor->call_count, 1);
    ASSERT_TRUE(first.find("cached") != first.end());
    ASSERT_EQ(first.at("cached").size(), 1u);
    EXPECT_EQ(first.at("cached")[0], second.at("cached")[0]);
}

TEST(doc_node, doc_path_reads_from_root_global_metadata) {
    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    (*global_meta)[std::string(lazyllm::RAGMetadataKeys::DOC_PATH)] = std::string("/tmp/a.txt");
    lazyllm::DocNode root("root", "", "root", nullptr, {}, global_meta);
    lazyllm::DocNode child("child", "", "child", &root, {}, global_meta);

    EXPECT_EQ(child.get_doc_path(), "/tmp/a.txt");

    child.set_doc_path("/tmp/b.txt");
    EXPECT_EQ(root.get_doc_path(), "/tmp/b.txt");
}

TEST(doc_node, embedding_keys_undone_throws_on_empty_input) {
    lazyllm::DocNode node;
    EXPECT_THROW(node.embedding_keys_undone({}), std::runtime_error);

    node.set_embedding_vec("done", std::vector<double>{1.0, 2.0});
    const auto missing = node.embedding_keys_undone({"done", "todo"});
    EXPECT_EQ(missing, (std::set<std::string>{"todo"}));
}

TEST(doc_node, py_do_embedding_writes_embedding_vector) {
    lazyllm::DocNode node("text");

    node.py_do_embedding({
        {"len_embedding", [](const std::string& input) {
            return std::vector<double>{static_cast<double>(input.size())};
        }}
    });

    ASSERT_TRUE(node._embedding_vecs.find("len_embedding") != node._embedding_vecs.end());
    const auto& embedding = node._embedding_vecs.at("len_embedding");
    ASSERT_TRUE(std::holds_alternative<std::vector<double>>(embedding));
    const auto& dense = std::get<std::vector<double>>(embedding);
    ASSERT_EQ(dense.size(), 1u);
    EXPECT_EQ(dense[0], 4.0); // "text"
}

TEST(doc_node, equality_uses_uid) {
    lazyllm::DocNode lhs("left", "", "same_uid");
    lazyllm::DocNode rhs("right", "", "same_uid");
    EXPECT_TRUE(lhs == rhs);

    lazyllm::DocNode other("other", "", "other_uid");
    EXPECT_TRUE(lhs != other);
}

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace lazyllm {

constexpr const char kRagKbId[] = "kb_id";
constexpr const char kRagDocId[] = "docid";
constexpr const char kRagDocPath[] = "lazyllm_doc_path";

enum class MetadataMode {
    All,
    Embed,
    Llm,
    None,
};

class DocNode {
public:
    using Embedding = std::unordered_map<std::string, std::vector<float>>;
    using Metadata = std::unordered_map<std::string, std::string>;
    using Children = std::unordered_map<std::string, std::vector<DocNode*>>;
    using EmbeddingFn = std::function<std::vector<float>(const std::string&, const std::string&)>;

    DocNode();
    explicit DocNode(const std::string& text);
    DocNode(const DocNode& other);
    DocNode& operator=(const DocNode& other);
    virtual ~DocNode() = default;

    const std::string& uid() const;
    const std::string& group() const;
    void set_group(const std::string& group);

    bool content_is_list() const;
    const std::vector<std::string>& content_list() const;
    const std::string& content_text() const;

    void set_content(const std::string& text);
    void set_content(const std::vector<std::string>& lines);

    void set_text(const std::string& text);
    const std::string& get_text() const;
    virtual std::string get_text_with_metadata(MetadataMode mode) const;

    std::string content_hash() const;

    Embedding& embedding();
    const Embedding& embedding() const;
    void set_embedding(const Embedding& embed);

    std::vector<std::string> has_missing_embedding(const std::vector<std::string>& embed_keys) const;
    virtual void do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed);
    void set_embedding_value(const std::string& key, const std::vector<float>& value);
    void check_embedding_state(const std::string& embed_key) const;

    DocNode* parent();
    const DocNode* parent() const;
    void set_parent(DocNode* parent);

    Children& children();
    const Children& children() const;
    void set_children(const Children& children);

    DocNode* root_node();
    const DocNode* root_node() const;
    bool is_root_node() const;

    Metadata& metadata();
    const Metadata& metadata() const;
    void set_metadata(const Metadata& metadata);

    Metadata& global_metadata();
    const Metadata& global_metadata() const;
    void set_global_metadata(const Metadata& global_metadata);

    std::vector<std::string> excluded_embed_metadata_keys() const;
    void set_excluded_embed_metadata_keys(const std::vector<std::string>& keys);
    std::vector<std::string> excluded_llm_metadata_keys() const;
    void set_excluded_llm_metadata_keys(const std::vector<std::string>& keys);

    std::string docpath() const;
    void set_docpath(const std::string& path);

    std::string get_children_str() const;
    std::string get_parent_id() const;

    std::string to_string() const;
    bool operator==(const DocNode& other) const;
    bool operator!=(const DocNode& other) const;
    std::size_t hash() const;

    std::string get_metadata_str(MetadataMode mode = MetadataMode::All) const;
    virtual std::string get_content(MetadataMode mode = MetadataMode::Llm) const;

    DocNode with_score(double score) const;
    DocNode with_sim_score(double score) const;

    bool has_relevance_score() const;
    bool has_similarity_score() const;
    double relevance_score() const;
    double similarity_score() const;

protected:
    void invalidate_content_hash();

    std::string uid_;
    std::string group_;
    std::string text_;
    bool content_is_list_;
    std::vector<std::string> content_list_;
    Embedding embedding_;
    Metadata metadata_;
    Metadata global_metadata_;
    std::vector<std::string> excluded_embed_metadata_keys_;
    std::vector<std::string> excluded_llm_metadata_keys_;
    DocNode* parent_;
    Children children_;
    bool children_loaded_;
    mutable std::mutex embedding_mutex_;
    mutable std::set<std::string> embedding_state_;
    mutable std::string content_hash_;
    mutable bool content_hash_dirty_;
    double relevance_score_;
    bool has_relevance_score_;
    double similarity_score_;
    bool has_similarity_score_;
};

class QADocNode : public DocNode {
public:
    QADocNode(const std::string& query, const std::string& answer);
    QADocNode(const std::string& query, const std::string& answer, const std::string& uid,
              const std::string& group = std::string());
    const std::string& answer() const;
    std::string get_text_with_metadata(MetadataMode mode) const override;

private:
    std::string answer_;
};

class ImageDocNode : public DocNode {
public:
    ImageDocNode(const std::string& image_path);
    ImageDocNode(const std::string& image_path, const std::string& uid,
                 const std::string& group = std::string());

    const std::string& image_path() const;
    std::string get_content(MetadataMode mode = MetadataMode::Llm) const override;
    void do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed) override;
    std::string get_text_with_metadata(MetadataMode mode) const override;

private:
    std::string image_path_;
    std::string modality_;
};

} // namespace lazyllm

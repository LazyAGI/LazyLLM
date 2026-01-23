#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <optional>
#include <any>
#include <memory>

#include "utils.hpp"
#include "document_store.hpp"

namespace lazyllm {

enum class MetadataMode { ALL, EMBED, LLM, NONE };

class DocNode {
public:
    using Metadata = std::unordered_map<std::string, std::any>;
    using Children = std::unordered_map<std::string, std::vector<const DocNode*>>;
    using EmbeddingFun = std::function<std::vector<double>(const std::string&, const std::string&)>;
    using EmbeddingVec = std::unordered_map<std::string, std::vector<double>>;

    mutable std::set<std::string> embedding_state = {};
    double relevance_score = .0;
    double similarity_score = .0;

private:
    std::string _text = "";
    std::string _uid = "";
    std::string _group_name = "";
    std::string _parent_group_name = "";
    std::vector<std::string> _content;
    mutable std::string _content_hash = "";
    EmbeddingVec _embedding;
    Metadata _metadata;
    Metadata _global_metadata;
    std::shared_ptr<DocumentStore> _store;
    std::vector<std::string> _excluded_embed_metadata_keys;
    std::vector<std::string> _excluded_llm_metadata_keys;
    DocNode* _p_parent_node = nullptr;
    Children _children;
    bool _children_loaded = false;

public:
    DocNode() = default;
    explicit DocNode(
        const std::string &text = "",
        const std::vector<std::string> &content = {},
        const std::string &group_name = "",
        const std::string &parent_group_name = "",
        const DocNode *p_parent_node = nullptr,
        const std::shared_ptr<DocumentStore> p_store = nullptr,
        const EmbeddingVec &embedding_vec = {},
        const Metadata &metadata = {},
        const Metadata &global_metadata = {},
        const std::string &uid = ""
    ) :
        _text(text),
        _content(content),
        _uid(uid),
        _group_name(group_name),
        _parent_group_name(parent_group_name),
        _embedding(embedding_vec),
        _metadata(metadata),
        _global_metadata(global_metadata),
        _store(p_store),
        _parent_node(p_parent_node) {}
    DocNode(const DocNode&) = default;
    DocNode& operator=(const DocNode&) = default;
    virtual ~DocNode() = default;

    // Getter and Setter
    const std::string& uid() const { return _uid; }
    void set_uid(const std::string& uid) { _uid = uid; }
    const std::string& group() const { return _group; }
    void set_group(const std::string& group) { _group = group; }
    void set_text(const std::string& text){
        _text = text;
        _content.clear();
        _content_hash = "";
    }
    void set_content(const std::vector<std::any>& content);
    const std::string& get_text() const;
    virtual std::string get_text_with_metadata(MetadataMode mode) const;
    void set_store(std::shared_ptr<DocumentStore> store);
    const std::shared_ptr<DocumentStore>& store() const;

    std::string content_hash() const;

    EmbeddingVec& embedding();
    const EmbeddingVec& embedding() const;
    void set_embedding(const EmbeddingVec& embed);

    std::vector<std::string> has_missing_embedding(const std::vector<std::string>& embed_keys) const;
    virtual void do_embedding(const std::unordered_map<std::string, EmbeddingFun>& embed);
    void set_embedding_value(const std::string& key, const std::vector<float>& value);
    void check_embedding_state(const std::string& embed_key) const;

    std::shared_ptr<DocNode> parent();
    const std::shared_ptr<DocNode> parent() const;
    void set_parent(std::shared_ptr<DocNode> parent);

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

    std::string get_metadata_str(MetadataMode mode = MetadataMode::ALL) const;
    virtual std::string get_content(MetadataMode mode = MetadataMode::LLM) const;

    DocNode with_score(double score) const;
    DocNode with_sim_score(double score) const;

    bool has_relevance_score() const;
    bool has_similarity_score() const;
    double relevance_score() const;
    double similarity_score() const;

    void generateUUID() { _uid = GenerateUUID(); }


};

class QADocNode : public DocNode {
public:
    QADocNode(const std::string& query, const std::string& answer);
    QADocNode(const std::string& query, const std::string& answer, const std::string& uid,
              const std::string& group = std::string());
    const std::string& answer() const;
    std::string get_text_with_metadata(MetadataMode mode) const override;

private:
    std::string _answer;
};

class ImageDocNode : public DocNode {
public:
    ImageDocNode(const std::string& image_path);
    ImageDocNode(const std::string& image_path, const std::string& uid,
                 const std::string& group = std::string());

    const std::string& image_path() const;
    std::string get_content(MetadataMode mode = MetadataMode::LLM) const override;
    void do_embedding(const std::unordered_map<std::string, EmbeddingFun>& embed) override;
    std::string get_text_with_metadata(MetadataMode mode) const override;

private:
    std::string _image_path;
    std::string _modality;
};

} // namespace lazyllm

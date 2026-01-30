#pragma once

#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <xxhash.h>
#include <unordered_map>
#include <vector>
#include <variant>
#include <optional>
#include <any>
#include <memory>

#include "utils.hpp"

namespace lazyllm {

enum class MetadataMode { ALL, EMBED, LLM, NONE };

class DocNode {
public:
    using Metadata = std::unordered_map<std::string, std::any>;
    using Children = std::unordered_map<std::string, std::vector<const DocNode*>>;
    using EmbeddingFun = std::function<std::vector<double>(const std::string&, const std::string&)>;
    using EmbeddingVec = std::unordered_map<std::string, std::vector<double>>;

    DocNode* _p_parent_node = nullptr;
private:
    std::shared_ptr<std::string> _p_root_text = nullptr;
    std::string_view _text_view;
    std::string _group_name;
    std::string _uid;
    mutable size_t _text_hash = 0;

    Metadata _metadata;
    std::shared_ptr<Metadata> _p_global_metadata;
    std::vector<std::string> _excluded_embed_metadata_keys;
    std::vector<std::string> _excluded_llm_metadata_keys;

    EmbeddingVec _embedding_vec;
    mutable std::set<std::string> _embedding_state = {};
    double _relevance_score = .0;
    double _similarity_score = .0;

    Children _children;

public:
    DocNode() = delete;
    explicit DocNode(
        const std::string_view& text_view,
        const std::string& group_name,
        const std::string& uid = "",
        const EmbeddingVec& embedding_vec = {},
        const Metadata& metadata = {},
        const std::shared_ptr<Metadata>& global_metadata = {},
        const std::shared_ptr<std::string>& p_raw_text = nullptr
    ) :
        _text_view(text_view),
        _group_name(group_name),
        _uid(uid),
        _metadata(metadata),
        _p_global_metadata(global_metadata),
        _embedding_vec(embedding_vec),
        _p_root_text(p_raw_text)
    {
        if (uid.empty()) _uid = GenerateUUID();
    }

    DocNode(const DocNode&) = default;
    DocNode& operator=(const DocNode&) = default;
    virtual ~DocNode() = default;

    size_t evaluate_text_hash() const {
        return static_cast<size_t>(XXH64(_text_view.data(), _text_view.size(), 0));
    }
    std::vector<DocNode*> find_children(const std::vector<DocNode*>& nodes) const {
        std::vector<DocNode*> children;
        for (auto p_node : nodes)
            if (p_node->get_parent_node() == _p_parent_node)
                children.push_back(p_node);
        return children;
    }

    // Getter and Setter
    const std::string& get_uid() const { return _uid; }
    const std::string& get_group_name() const { return _group_name; }
    const std::string_view& get_text_view() const { return _text_view; }
    const std::string& get_text() const { return std::string(_text_view); }
    void set_text(const std::string& text) {
        _p_root_text = std::make_shared<std::string>(text);
        _text_view = *_p_root_text;
        _text_hash = evaluate_text_hash();
    }
    const size_t& get_text_hash() const {
        if (_text_hash == 0) _text_hash = evaluate_text_hash();
        return _text_hash;
    }
    const DocNode* get_parent_node() { return _p_parent_node; }
    void set_parent_node(DocNode* p_parent_node) { _p_parent_node = p_parent_node; }

    const std::vector<DocNode*>& get_children() const {
        return _children;
    }

    const Metadata& get_metadata() const { return _metadata; }
    void set_metadata(const Metadata& metadata) { _metadata = metadata; }

    std::shared_ptr<Metadata> global_metadata_ptr() { return _p_global_metadata; }
    std::shared_ptr<const Metadata> global_metadata_ptr() const { return _p_global_metadata; }
    Metadata& global_metadata() { return *_p_global_metadata; }
    const Metadata& global_metadata() const { return *_p_global_metadata; }
    void set_global_metadata_ptr(std::shared_ptr<Metadata> p_global_metadata) {
        _p_global_metadata = std::move(p_global_metadata);
    }
    void set_global_metadata(const Metadata& global_metadata) {
        _p_global_metadata = std::make_shared<Metadata>(global_metadata);
    }

    const std::vector<std::string>& excluded_embed_metadata_keys() const {
        return _excluded_embed_metadata_keys;
    }
    void set_excluded_embed_metadata_keys(const std::vector<std::string>& keys) {
        _excluded_embed_metadata_keys = keys;
    }

    const std::vector<std::string>& excluded_llm_metadata_keys() const {
        return _excluded_llm_metadata_keys;
    }
    void set_excluded_llm_metadata_keys(const std::vector<std::string>& keys) {
        _excluded_llm_metadata_keys = keys;
    }

    EmbeddingVec& embedding_vec() { return _embedding_vec; }
    const EmbeddingVec& embedding_vec() const { return _embedding_vec; }
    void set_embedding_vec(const EmbeddingVec& embedding_vec) { _embedding_vec = embedding_vec; }

    std::set<std::string>& embedding_state() { return _embedding_state; }
    const std::set<std::string>& embedding_state() const { return _embedding_state; }
    void set_embedding_state(const std::set<std::string>& embedding_state) {
        _embedding_state = embedding_state;
    }

    double relevance_score() const { return _relevance_score; }
    void set_relevance_score(double relevance_score) { _relevance_score = relevance_score; }

    double similarity_score() const { return _similarity_score; }
    void set_similarity_score(double similarity_score) { _similarity_score = similarity_score; }

    const std::variant<std::string, DocNode*>& parent_node() const { return _p_parent_node; }
    void set_parent_node(const std::variant<std::string, DocNode*>& parent_node) {
        _p_parent_node = parent_node;
    }

    Children& children() { return _children; }
    const Children& children() const { return _children; }
    void set_children(const Children& children) { _children = children; }

    bool children_loaded() const { return _children_loaded; }
    void set_children_loaded(bool children_loaded) { _children_loaded = children_loaded; }
};

} // namespace lazyllm

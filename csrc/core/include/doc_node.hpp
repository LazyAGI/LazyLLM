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
#include "adaptor_base.hpp"

namespace lazyllm {

enum class MetadataMode { ALL, EMBED, LLM, NONE };

class DocNode {
public:
    using Metadata = std::unordered_map<std::string, std::any>;
    using Children = std::unordered_map<std::string, std::vector<DocNode*>>;
    using EmbeddingFun = std::function<std::vector<double>(const std::string&, const std::string&)>;
    using EmbeddingVec = std::unordered_map<std::string, std::vector<double>>;

    Metadata _metadata;
    std::shared_ptr<Metadata> _p_global_metadata;
    EmbeddingVec _embedding_vec;
private:
    std::string_view _text_view;
    std::shared_ptr<std::string> _p_root_text = nullptr;
    std::vector<std::string> _root_texts = {};
    std::string _group_name;
    std::string _uid;
    mutable size_t _text_hash = 0;

    std::vector<std::string> _excluded_embed_metadata_keys;
    std::vector<std::string> _excluded_llm_metadata_keys;

    mutable std::set<std::string> _embedding_state = {};
    double _relevance_score = .0;
    double _similarity_score = .0;

    const DocNode* _p_parent_node = nullptr;
    mutable Children _children;
    std::shared_ptr<const AdaptorBase> _p_store = nullptr;

public:
    DocNode() = delete;
    explicit DocNode(
        const std::string_view& text_view,
        const std::string& group_name,
        const std::string& uid = "",
        const DocNode* p_parent_node = nullptr,
        const Metadata& metadata = {},
        const std::shared_ptr<Metadata>& global_metadata = {}
    ) :
        _group_name(group_name),
        _uid(uid.empty() ? GenerateUUID() : uid),
        _p_parent_node(p_parent_node),
        _metadata(metadata),
        _p_global_metadata(global_metadata)
    {
        set_text_view(text_view);
    }

    DocNode(const DocNode&) = default;
    DocNode& operator=(const DocNode&) = default;
    virtual ~DocNode() = default;

    size_t evaluate_text_hash() const {
        return static_cast<size_t>(XXH64(_text_view.data(), _text_view.size(), 0));
    }

    // Getter and Setter
    void set_store(const std::shared_ptr<AdaptorBase>& p_store) { _p_store = p_store; }
    const std::string& get_uid() const { return _uid; }
    const std::string& get_group_name() const { return _group_name; }
    const std::string_view& get_text_view() const { return _text_view; }
    void set_text_view(const std::string_view& text_view) {
        _text_view = text_view;
        _text_hash = evaluate_text_hash();
    }
    const std::string& get_text() const { return std::string(_text_view); }
    void set_root_text(const std::string&& text) {
        _p_root_text = std::make_shared<std::string>(std::move(text));
        set_text_view(*_p_root_text);
    }
    void set_root_texts(const std::vector<std::string>& texts) { set_root_text(JoinLines(texts)); }
    size_t text_hash() const { return _text_hash; }
    const DocNode* get_parent_node() { return _p_parent_node; }
    void set_parent_node(DocNode* p_parent_node) { _p_parent_node = p_parent_node; }
    const Children& get_children() const {
        if (!_children.empty()) return _children;
        if (_p_store == nullptr) return Children();
        _children = std::any_cast<Children>(_p_store->call("get_node_children", {{"node", this}}));
        return _children;
    }
    void set_children(const Children& children) { _children = children; }




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

};

} // namespace lazyllm

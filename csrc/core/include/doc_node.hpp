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
    using EmbeddingVecs = std::unordered_map<std::string, std::vector<double>>;

    Metadata _metadata;
    std::shared_ptr<Metadata> _p_global_metadata;
    EmbeddingVecs _embedding_vecs;
    std::set<std::string> _pending_embedding_keys = {};
    double _relevance_score = .0;
    double _similarity_score = .0;

private:
    std::string_view _text_view;
    std::shared_ptr<std::string> _p_root_text = nullptr;
    std::vector<std::string> _root_texts = {};
    std::string _group_name;
    std::string _uid;
    mutable size_t _text_hash = 0;

    std::set<std::string> _excluded_embed_metadata_keys;
    std::set<std::string> _excluded_llm_metadata_keys;

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
    const DocNode* get_root_node() const {
        if (_p_parent_node == nullptr) return this;
        return _p_parent_node->get_root_node();
    }
    void set_store(const std::shared_ptr<AdaptorBase>& p_store) { _p_store = p_store; }
    const std::string& get_uid() const { return _uid; }
    const std::string& get_group_name() const { return _group_name; }
    const std::string_view& get_text_view() const { return _text_view; }
    void set_text_view(const std::string_view& text_view) {
        _text_view = text_view;
        _text_hash = evaluate_text_hash();
    }
    std::string get_metadata_string(MetadataMode mode = MetadataMode::ALL) const {
        if (mode == MetadataMode::NONE) return "";

        std::set<std::string> valid_keys;
        for (const auto& [key, _] : _metadata) valid_keys.insert(key);

        if (mode == MetadataMode::LLM)
            valid_keys = SetDiff(valid_keys, _excluded_llm_metadata_keys);
        else if (mode == MetadataMode::EMBED)
            valid_keys = SetDiff(valid_keys, _excluded_embed_metadata_keys);

        std::vector<std::string> kv_strings;
        for (const std::string& key : valid_keys)
            kv_strings.emplace_back(key + ":" + std::any_cast<std::string>(_metadata.at(key)));

        return JoinLines(kv_strings);
    }
    std::string get_text(MetadataMode mode = MetadataMode::NONE) const {
        if (mode == MetadataMode::NONE) return std::string(_text_view);
        const auto& metadata_string = get_metadata_string(mode);
        return metadata_string + "\n\n" + std::string(_text_view);
    }
    void set_root_text(const std::string&& text) {
        _p_root_text = std::make_shared<std::string>(std::move(text));
        set_text_view(*_p_root_text);
    }
    void set_root_texts(const std::vector<std::string>& texts) { set_root_text(JoinLines(texts)); }
    size_t text_hash() const { return _text_hash; }
    const DocNode* get_parent_node() const { return _p_parent_node; }
    void set_parent_node(DocNode* p_parent_node) { _p_parent_node = p_parent_node; }
    Children py_get_children() const {
        if (!_children.empty()) return _children;
        if (_p_store == nullptr) return Children();
        _children = std::any_cast<Children>(_p_store->call("get_node_children", {{"node", this}}));
        return _children;
    }
    void set_children(const Children& children) { _children = children; }
    std::set<std::string> get_excluded_embed_metadata_keys() const {
        return SetUnion(get_root_node()->get_excluded_embed_metadata_keys(), _excluded_embed_metadata_keys);
    }
    void set_excluded_embed_metadata_keys(const std::set<std::string>& keys) {
        _excluded_embed_metadata_keys = keys;
    }
    std::set<std::string> get_excluded_llm_metadata_keys() const {
        return SetUnion(get_root_node()->get_excluded_llm_metadata_keys(), _excluded_llm_metadata_keys);
    }
    void set_excluded_llm_metadata_keys(const std::set<std::string>& keys) {
        _excluded_llm_metadata_keys = keys;
    }
    std::string get_doc_path() const {
        return std::any_cast<std::string>(get_root_node()->_p_global_metadata->at(std::string(RAG_KEY_DOC_PATH)));
    }
    void set_doc_path(const std::string& path) {
        get_root_node()->_p_global_metadata->operator[](std::string(RAG_KEY_DOC_PATH)) = path;
    }
    auto py_get_children_uid() const {
        auto children = py_get_children();
        std::unordered_map<std::string, std::vector<std::string>> children_uid;
        for (auto& [group_name, nodes] : children) {
            children_uid[group_name] = {};
            for (auto& node : nodes)
                children_uid[group_name].push_back(node->get_uid());
        }
        return children_uid;
    }
    std::string get_parent_uid() const {
        auto parent = get_parent_node();
        if (parent == nullptr) return "";
        return parent->get_uid();
    }
    std::set<std::string> embedding_keys_undone(const std::set<std::string>& keys_done) const {
        if (keys_done.empty()) throw std::runtime_error("The ebmed_keys to be checked must be passed in.");;
        std::set<std::string> keys_undone;
        for (const auto& key : keys_done) {
            if (_embedding_vecs.find(key) == _embedding_vecs.end())
                keys_undone.insert(key);
        }
        return keys_undone;
    }
    void py_do_embedding(const std::unordered_map<std::string,
        std::function<std::vector<double>(const std::string&)>>& embedding_funcs) {
        for (const auto& [key, func] : embedding_funcs)
            _embedding_vecs[key] = func(get_text(MetadataMode::EMBED));
    }
    void set_embedding_vec(const std::string& key, const std::vector<double>& embedding_vec) {
        _embedding_vecs[key] = embedding_vec;
    }

    bool operator==(const DocNode& other) const { return _uid == other._uid; }
    bool operator!=(const DocNode& other) const { return _uid != other._uid; }
};

} // namespace lazyllm

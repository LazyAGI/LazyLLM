#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <optional>

namespace lazyllm {

enum class MetadataMode { ALL, EMBED, LLM, NONE };

class DocNode {
protected:
    using Metadata = std::unordered_map<std::string, std::string>;
    using Children = std::unordered_map<std::string, std::vector<DocNode*>>;
    using EmbeddingFun = std::function<std::vector<float>(const std::string&, const std::string&)>;
    using EmbeddingVec = std::unordered_map<std::string, std::vector<float>>;

public:
    DocNode(
        uid: Optional[str] = None,
        content: Optional[Union[str, List[Any]]] = None,
        group: Optional[str] = None,
        embedding: Optional[Dict[str,List[float]]] = None,
        parent: Optional[Union[str, 'DocNode']] = None,
        store=None,
        node_groups: Optional[Dict[str, Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        global_metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    );
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

    EmbeddingVec& embedding();
    const EmbeddingVec& embedding() const;
    void set_embedding(const EmbeddingVec& embed);

    std::vector<std::string> has_missing_embedding(const std::vector<std::string>& embed_keys) const;
    virtual void do_embedding(const std::unordered_map<std::string, EmbeddingFun>& embed);
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

    std::string get_metadata_str(MetadataMode mode = MetadataMode::ALL) const;
    virtual std::string get_content(MetadataMode mode = MetadataMode::LLM) const;

    DocNode with_score(double score) const;
    DocNode with_sim_score(double score) const;

    bool has_relevance_score() const;
    bool has_similarity_score() const;
    double relevance_score() const;
    double similarity_score() const;

protected:
    void invalidate_content_hash();

    std::string _uid;
    std::string _group;
    std::string _text;
    bool _content_is_list;
    std::vector<std::string> _content_list;
    EmbeddingVec _embedding;
    Metadata _metadata;
    Metadata _global_metadata;
    std::vector<std::string> _excluded_embed_metadata_keys;
    std::vector<std::string> _excluded_llm_metadata_keys;
    DocNode* _parent;
    Children _children;
    bool _children_loaded;
    mutable std::set<std::string> _embedding_state;
    mutable std::string _content_hash;
    mutable bool _content_hash_dirty;
    double _relevance_score;
    bool _has_relevance_score;
    double _similarity_score;
    bool _has_similarity_score;
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

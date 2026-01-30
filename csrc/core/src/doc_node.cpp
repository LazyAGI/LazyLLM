#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "doc_node.hpp"
#include "utils.hpp"

namespace lazyllm {

const std::string& DocNode::content_text() const {
    return _text;
}

void DocNode::set_content(const std::vector<std::string>& lines) {
    _content_is_list = true;
    _content = lines;
    _text = JoinLines(lines);
    content_hash();
}



const std::string& DocNode::get_text() const {
    return _text_view;
}

std::string DocNode::get_text_with_metadata(MetadataMode mode) const {
    const std::string metadata_str = get_metadata_str(mode);
    if (metadata_str.empty()) {
        return _text;
    }
    if (_text.empty()) {
        return metadata_str;
    }
    return metadata_str + "\n\n" + _text;
}

void DocNode::set_store(std::shared_ptr<DocumentStore> store) {
    _store = std::move(store);
}

const std::shared_ptr<DocumentStore>& DocNode::store() const {
    return _store;
}

std::string DocNode::content_hash() const {
    _content_hash = Sha256Hex(_text);
    return _content_hash;
}

DocNode::EmbeddingVec& DocNode::embedding() {
    return _embedding;
}

const DocNode::EmbeddingVec& DocNode::embedding() const {
    return _embedding;
}

void DocNode::set_embedding(const EmbeddingVec& embed) {
    std::lock_guard<std::mutex> lock(_embedding_mutex);
    _embedding = embed;
}

std::vector<std::string> DocNode::has_missing_embedding(const std::vector<std::string>& embed_keys) const {
    std::vector<std::string> missing;
    if (embed_keys.empty()) {
        return missing;
    }
    std::lock_guard<std::mutex> lock(_embedding_mutex);
    for (const auto& key : embed_keys) {
        if (_embedding.find(key) == _embedding.end()) {
            missing.push_back(key);
        }
    }
    return missing;
}

void DocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFun>& embed) {
    EmbeddingVec generated;
    const std::string input = get_text_with_metadata(MetadataMode::EMBED);
    for (const auto& item : embed) {
        generated[item.first] = item.second(input, "");
    }
    for (const auto& item : generated) {
        _embedding[item.first] = item.second;
    }
}

void DocNode::set_embedding_value(const std::string& key, const std::vector<float>& value) {
    _embedding[key] = value;
}

void DocNode::check_embedding_state(const std::string& embed_key) const {
    while (true) {
        {
            if (_embedding.find(embed_key) != _embedding.end()) {
                _embedding_state.erase(embed_key);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

DocNode* DocNode::parent() {
    return _parent;
}

const DocNode* DocNode::parent() const {
    return _parent;
}

void DocNode::set_parent(DocNode* parent) {
    _parent = parent;
}

DocNode::Children& DocNode::children() {
    return _children;
}

const DocNode::Children& DocNode::children() const {
    return _children;
}

void DocNode::set_children(const Children& children) {
    _children = children;
}

DocNode* DocNode::root_node() {
    DocNode* node = this;
    while (node->_parent != nullptr) {
        node = node->_parent;
    }
    return node;
}

const DocNode* DocNode::root_node() const {
    const DocNode* node = this;
    while (node->_parent != nullptr) {
        node = node->_parent;
    }
    return node;
}

bool DocNode::is_root_node() const {
    return _parent == nullptr;
}

DocNode::Metadata& DocNode::metadata() {
    return _metadata;
}

const DocNode::Metadata& DocNode::metadata() const {
    return _metadata;
}

void DocNode::set_metadata(const Metadata& metadata) {
    _metadata = metadata;
}

DocNode::Metadata& DocNode::global_metadata() {
    return root_node()->_global_metadata;
}

const DocNode::Metadata& DocNode::global_metadata() const {
    return root_node()->_global_metadata;
}

void DocNode::set_global_metadata(const Metadata& global_metadata) {
    _global_metadata = global_metadata;
}

std::vector<std::string> DocNode::excluded_embed_metadata_keys() const {
    std::set<std::string> keys;
    const DocNode* root = root_node();
    keys.insert(root->_excluded_embed_metadata_keys.begin(), root->_excluded_embed_metadata_keys.end());
    keys.insert(_excluded_embed_metadata_keys.begin(), _excluded_embed_metadata_keys.end());
    return std::vector<std::string>(keys.begin(), keys.end());
}

void DocNode::set_excluded_embed_metadata_keys(const std::vector<std::string>& keys) {
    _excluded_embed_metadata_keys = keys;
}

std::vector<std::string> DocNode::excluded_llm_metadata_keys() const {
    std::set<std::string> keys;
    const DocNode* root = root_node();
    keys.insert(root->_excluded_llm_metadata_keys.begin(), root->_excluded_llm_metadata_keys.end());
    keys.insert(_excluded_llm_metadata_keys.begin(), _excluded_llm_metadata_keys.end());
    return std::vector<std::string>(keys.begin(), keys.end());
}

void DocNode::set_excluded_llm_metadata_keys(const std::vector<std::string>& keys) {
    _excluded_llm_metadata_keys = keys;
}

std::string DocNode::docpath() const {
    const auto& meta = global_metadata();
    const auto it = meta.find(kRagDocPath);
    if (it == meta.end()) {
        return "";
    }
    return it->second;
}

void DocNode::set_docpath(const std::string& path) {
    if (!is_root_node()) {
        throw std::runtime_error("Only root node can set docpath.");
    }
    global_metadata()[kRagDocPath] = path;
}

std::string DocNode::get_children_str() const {
    std::ostringstream oss;
    oss << "{";
    bool first_group = true;
    for (const auto& item : _children) {
        if (!first_group) {
            oss << ", ";
        }
        first_group = false;
        oss << item.first << ": [";
        bool first_child = true;
        for (const auto* node : item.second) {
            if (!node) {
                continue;
            }
            if (!first_child) {
                oss << ", ";
            }
            first_child = false;
            oss << node->uid();
        }
        oss << "]";
    }
    oss << "}";
    return oss.str();
}

std::string DocNode::get_parent_id() const {
    return _parent ? _parent->uid() : "";
}

std::string DocNode::to_string() const {
    std::ostringstream oss;
    oss << "DocNode(id: " << _uid << ", group: " << _group << ", content: " << _text << ") parent: "
        << get_parent_id() << ", children: " << get_children_str();
    return oss.str();
}

bool DocNode::operator==(const DocNode& other) const {
    return _uid == other._uid;
}

bool DocNode::operator!=(const DocNode& other) const {
    return !(*this == other);
}

std::size_t DocNode::hash() const {
    return std::hash<std::string>()(_uid);
}

std::string DocNode::get_metadata_str(MetadataMode mode) const {
    if (mode == MetadataMode::NONE) {
        return "";
    }
    std::set<std::string> keys;
    for (const auto& item : _metadata) {
        keys.insert(item.first);
    }
    if (mode == MetadataMode::LLM) {
        const auto excluded = excluded_llm_metadata_keys();
        for (const auto& key : excluded) {
            keys.erase(key);
        }
    } else if (mode == MetadataMode::EMBED) {
        const auto excluded = excluded_embed_metadata_keys();
        for (const auto& key : excluded) {
            keys.erase(key);
        }
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto& key : keys) {
        const auto it = _metadata.find(key);
        if (it == _metadata.end()) {
            continue;
        }
        if (!first) {
            oss << "\n";
        }
        first = false;
        oss << key << ": " << it->second;
    }
    return oss.str();
}

std::string DocNode::get_content(MetadataMode mode) const {
    if (mode == MetadataMode::LLM) {
        return get_text_with_metadata(MetadataMode::LLM);
    }
    return get_text_with_metadata(mode);
}

DocNode DocNode::with_score(double score) const {
    DocNode node(*this);
    node._relevance_score = score;
    return node;
}

DocNode DocNode::with_sim_score(double score) const {
    DocNode node(*this);
    node._similarity_score = score;
    return node;
}

double DocNode::relevance_score() const {
    return _relevance_score;
}

double DocNode::similarity_score() const {
    return _similarity_score;
}

QADocNode::QADocNode(const std::string& query, const std::string& answer)
    : DocNode(query), _answer(Trim(answer)) {}

QADocNode::QADocNode(const std::string& query, const std::string& answer, const std::string& uid,
                     const std::string& group)
    : DocNode(query), _answer(Trim(answer)) {
    if (!uid.empty()) {
        _uid = uid;
    }
    _group = group;
}

const std::string& QADocNode::answer() const {
    return _answer;
}

std::string QADocNode::get_text_with_metadata(MetadataMode mode) const {
    if (mode == MetadataMode::LLM) {
        std::ostringstream oss;
        oss << "query:\n" << _text << "\nanswer\n" << _answer;
        return oss.str();
    }
    return DocNode::get_text_with_metadata(mode);
}

ImageDocNode::ImageDocNode(const std::string& image_path)
    : DocNode(image_path), _image_path(Trim(image_path)), _modality("image") {
    set_text(_image_path);
}

ImageDocNode::ImageDocNode(const std::string& image_path, const std::string& uid, const std::string& group)
    : DocNode(image_path), _image_path(Trim(image_path)), _modality("image") {
    if (!uid.empty()) {
        _uid = uid;
    }
    _group = group;
    set_text(_image_path);
}

const std::string& ImageDocNode::image_path() const {
    return _image_path;
}

std::string ImageDocNode::get_content(MetadataMode mode) const {
    if (mode == MetadataMode::EMBED) {
        std::string file_bytes;
        if (!ReadFileBinary(_image_path, &file_bytes)) {
            return "";
        }
        const std::string mime = ImageMimeType(_image_path);
        const std::string base64 = Base64Encode(file_bytes);
        if (mime.empty()) {
            return base64;
        }
        return "data:" + mime + ";base64," + base64;
    }
    return _image_path;
}

void ImageDocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFun>& embed) {
    EmbeddingVec generated;
    const std::string input = get_content(MetadataMode::EMBED);
    for (const auto& item : embed) {
        generated[item.first] = item.second(input, _modality);
    }
    std::lock_guard<std::mutex> lock(_embedding_mutex);
    for (const auto& item : generated) {
        _embedding[item.first] = item.second;
    }
}

std::string ImageDocNode::get_text_with_metadata(MetadataMode mode) const {
    (void)mode;
    return _image_path;
}

} // namespace lazyllm

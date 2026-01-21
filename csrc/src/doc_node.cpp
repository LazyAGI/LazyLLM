#include "doc_node.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace lazyllm {
namespace {

std::string JoinLines(const std::vector<std::string>& lines) {
    std::ostringstream oss;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            oss << "\n";
        }
        oss << lines[i];
    }
    return oss.str();
}

std::string Trim(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string GenerateUUID() {
    static const char kHex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    const int groups[] = {8, 4, 4, 4, 12};
    std::string out;
    out.reserve(36);
    for (size_t group = 0; group < 5; ++group) {
        if (group > 0) {
            out.push_back('-');
        }
        for (int i = 0; i < groups[group]; ++i) {
            out.push_back(kHex[dist(gen)]);
        }
    }
    return out;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string GetExtension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= path.size()) {
        return "";
    }
    return ToLower(path.substr(dot + 1));
}

std::string ImageMimeType(const std::string& path) {
    const std::string ext = GetExtension(path);
    if (ext == "jpg" || ext == "jpeg" || ext == "jfif" || ext == "jpe") {
        return "image/jpeg";
    }
    if (ext == "png" || ext == "apng") {
        return "image/png";
    }
    if (ext == "gif") {
        return "image/gif";
    }
    if (ext == "bmp" || ext == "dib") {
        return "image/bmp";
    }
    if (ext == "tif" || ext == "tiff") {
        return "image/tiff";
    }
    if (ext == "webp") {
        return "image/webp";
    }
    if (ext == "ico") {
        return "image/x-icon";
    }
    if (ext == "icns") {
        return "image/icns";
    }
    return "";
}

bool ReadFileBinary(const std::string& path, std::string* out) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    *out = buffer.str();
    return true;
}

std::string Base64Encode(const std::string& data) {
    static const char kBase64Chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    int val = 0;
    int valb = -6;
    for (unsigned char c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(kBase64Chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        out.push_back(kBase64Chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (out.size() % 4) {
        out.push_back('=');
    }
    return out;
}

uint32_t RotateRight(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

std::string Sha256Hex(const std::string& input) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    };
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    std::vector<uint8_t> msg(input.begin(), input.end());
    const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8;
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF));
    }

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            const size_t idx = offset + i * 4;
            w[i] = (static_cast<uint32_t>(msg[idx]) << 24) |
                   (static_cast<uint32_t>(msg[idx + 1]) << 16) |
                   (static_cast<uint32_t>(msg[idx + 2]) << 8) |
                   (static_cast<uint32_t>(msg[idx + 3]));
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];
        uint32_t e = h[4];
        uint32_t f = h[5];
        uint32_t g = h[6];
        uint32_t h0 = h[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h0 + s1 + ch + k[i] + w[i];
            const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;

            h0 = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h0;
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < 8; ++i) {
        oss << std::setw(8) << h[i];
    }
    return oss.str();
}

} // namespace

DocNode::DocNode()
    : uid_(GenerateUUID()),
      group_(),
      text_(),
      content_is_list_(false),
      parent_(nullptr),
      children_loaded_(false),
      content_hash_(),
      content_hash_dirty_(true),
      relevance_score_(0.0),
      has_relevance_score_(false),
      similarity_score_(0.0),
      has_similarity_score_(false) {}

DocNode::DocNode(const std::string& text) : DocNode() {
    set_text(text);
}

DocNode::DocNode(const DocNode& other)
    : uid_(other.uid_),
      group_(other.group_),
      text_(other.text_),
      content_is_list_(other.content_is_list_),
      content_list_(other.content_list_),
      embedding_(other.embedding_),
      metadata_(other.metadata_),
      global_metadata_(other.global_metadata_),
      excluded_embed_metadata_keys_(other.excluded_embed_metadata_keys_),
      excluded_llm_metadata_keys_(other.excluded_llm_metadata_keys_),
      parent_(other.parent_),
      children_(other.children_),
      children_loaded_(other.children_loaded_),
      embedding_state_(other.embedding_state_),
      content_hash_(other.content_hash_),
      content_hash_dirty_(other.content_hash_dirty_),
      relevance_score_(other.relevance_score_),
      has_relevance_score_(other.has_relevance_score_),
      similarity_score_(other.similarity_score_),
      has_similarity_score_(other.has_similarity_score_) {}

DocNode& DocNode::operator=(const DocNode& other) {
    if (this == &other) {
        return *this;
    }
    uid_ = other.uid_;
    group_ = other.group_;
    text_ = other.text_;
    content_is_list_ = other.content_is_list_;
    content_list_ = other.content_list_;
    embedding_ = other.embedding_;
    metadata_ = other.metadata_;
    global_metadata_ = other.global_metadata_;
    excluded_embed_metadata_keys_ = other.excluded_embed_metadata_keys_;
    excluded_llm_metadata_keys_ = other.excluded_llm_metadata_keys_;
    parent_ = other.parent_;
    children_ = other.children_;
    children_loaded_ = other.children_loaded_;
    embedding_state_ = other.embedding_state_;
    content_hash_ = other.content_hash_;
    content_hash_dirty_ = other.content_hash_dirty_;
    relevance_score_ = other.relevance_score_;
    has_relevance_score_ = other.has_relevance_score_;
    similarity_score_ = other.similarity_score_;
    has_similarity_score_ = other.has_similarity_score_;
    return *this;
}

const std::string& DocNode::uid() const {
    return uid_;
}

const std::string& DocNode::group() const {
    return group_;
}

void DocNode::set_group(const std::string& group) {
    group_ = group;
}

bool DocNode::content_is_list() const {
    return content_is_list_;
}

const std::vector<std::string>& DocNode::content_list() const {
    return content_list_;
}

const std::string& DocNode::content_text() const {
    return text_;
}

void DocNode::set_content(const std::string& text) {
    set_text(text);
}

void DocNode::set_content(const std::vector<std::string>& lines) {
    content_is_list_ = true;
    content_list_ = lines;
    text_ = JoinLines(lines);
    invalidate_content_hash();
}

void DocNode::set_text(const std::string& text) {
    text_ = text;
    content_is_list_ = false;
    content_list_.clear();
    invalidate_content_hash();
}

const std::string& DocNode::get_text() const {
    return text_;
}

std::string DocNode::get_text_with_metadata(MetadataMode mode) const {
    const std::string metadata_str = get_metadata_str(mode);
    if (metadata_str.empty()) {
        return text_;
    }
    if (text_.empty()) {
        return metadata_str;
    }
    return metadata_str + "\n\n" + text_;
}

std::string DocNode::content_hash() const {
    if (content_hash_dirty_) {
        content_hash_ = Sha256Hex(text_);
        content_hash_dirty_ = false;
    }
    return content_hash_;
}

DocNode::Embedding& DocNode::embedding() {
    return embedding_;
}

const DocNode::Embedding& DocNode::embedding() const {
    return embedding_;
}

void DocNode::set_embedding(const Embedding& embed) {
    std::lock_guard<std::mutex> lock(embedding_mutex_);
    embedding_ = embed;
}

std::vector<std::string> DocNode::has_missing_embedding(const std::vector<std::string>& embed_keys) const {
    std::vector<std::string> missing;
    if (embed_keys.empty()) {
        return missing;
    }
    std::lock_guard<std::mutex> lock(embedding_mutex_);
    for (const auto& key : embed_keys) {
        if (embedding_.find(key) == embedding_.end()) {
            missing.push_back(key);
        }
    }
    return missing;
}

void DocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed) {
    Embedding generated;
    const std::string input = get_text_with_metadata(MetadataMode::Embed);
    for (const auto& item : embed) {
        generated[item.first] = item.second(input, "");
    }
    std::lock_guard<std::mutex> lock(embedding_mutex_);
    for (const auto& item : generated) {
        embedding_[item.first] = item.second;
    }
}

void DocNode::set_embedding_value(const std::string& key, const std::vector<float>& value) {
    std::lock_guard<std::mutex> lock(embedding_mutex_);
    embedding_[key] = value;
}

void DocNode::check_embedding_state(const std::string& embed_key) const {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(embedding_mutex_);
            if (embedding_.find(embed_key) != embedding_.end()) {
                embedding_state_.erase(embed_key);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

DocNode* DocNode::parent() {
    return parent_;
}

const DocNode* DocNode::parent() const {
    return parent_;
}

void DocNode::set_parent(DocNode* parent) {
    parent_ = parent;
}

DocNode::Children& DocNode::children() {
    return children_;
}

const DocNode::Children& DocNode::children() const {
    return children_;
}

void DocNode::set_children(const Children& children) {
    children_ = children;
}

DocNode* DocNode::root_node() {
    DocNode* node = this;
    while (node->parent_ != nullptr) {
        node = node->parent_;
    }
    return node;
}

const DocNode* DocNode::root_node() const {
    const DocNode* node = this;
    while (node->parent_ != nullptr) {
        node = node->parent_;
    }
    return node;
}

bool DocNode::is_root_node() const {
    return parent_ == nullptr;
}

DocNode::Metadata& DocNode::metadata() {
    return metadata_;
}

const DocNode::Metadata& DocNode::metadata() const {
    return metadata_;
}

void DocNode::set_metadata(const Metadata& metadata) {
    metadata_ = metadata;
}

DocNode::Metadata& DocNode::global_metadata() {
    return root_node()->global_metadata_;
}

const DocNode::Metadata& DocNode::global_metadata() const {
    return root_node()->global_metadata_;
}

void DocNode::set_global_metadata(const Metadata& global_metadata) {
    global_metadata_ = global_metadata;
}

std::vector<std::string> DocNode::excluded_embed_metadata_keys() const {
    std::set<std::string> keys;
    const DocNode* root = root_node();
    keys.insert(root->excluded_embed_metadata_keys_.begin(), root->excluded_embed_metadata_keys_.end());
    keys.insert(excluded_embed_metadata_keys_.begin(), excluded_embed_metadata_keys_.end());
    return std::vector<std::string>(keys.begin(), keys.end());
}

void DocNode::set_excluded_embed_metadata_keys(const std::vector<std::string>& keys) {
    excluded_embed_metadata_keys_ = keys;
}

std::vector<std::string> DocNode::excluded_llm_metadata_keys() const {
    std::set<std::string> keys;
    const DocNode* root = root_node();
    keys.insert(root->excluded_llm_metadata_keys_.begin(), root->excluded_llm_metadata_keys_.end());
    keys.insert(excluded_llm_metadata_keys_.begin(), excluded_llm_metadata_keys_.end());
    return std::vector<std::string>(keys.begin(), keys.end());
}

void DocNode::set_excluded_llm_metadata_keys(const std::vector<std::string>& keys) {
    excluded_llm_metadata_keys_ = keys;
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
    for (const auto& item : children_) {
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
    return parent_ ? parent_->uid() : "";
}

std::string DocNode::to_string() const {
    std::ostringstream oss;
    oss << "DocNode(id: " << uid_ << ", group: " << group_ << ", content: " << text_ << ") parent: "
        << get_parent_id() << ", children: " << get_children_str();
    return oss.str();
}

bool DocNode::operator==(const DocNode& other) const {
    return uid_ == other.uid_;
}

bool DocNode::operator!=(const DocNode& other) const {
    return !(*this == other);
}

std::size_t DocNode::hash() const {
    return std::hash<std::string>()(uid_);
}

std::string DocNode::get_metadata_str(MetadataMode mode) const {
    if (mode == MetadataMode::None) {
        return "";
    }
    std::set<std::string> keys;
    for (const auto& item : metadata_) {
        keys.insert(item.first);
    }
    if (mode == MetadataMode::Llm) {
        const auto excluded = excluded_llm_metadata_keys();
        for (const auto& key : excluded) {
            keys.erase(key);
        }
    } else if (mode == MetadataMode::Embed) {
        const auto excluded = excluded_embed_metadata_keys();
        for (const auto& key : excluded) {
            keys.erase(key);
        }
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto& key : keys) {
        const auto it = metadata_.find(key);
        if (it == metadata_.end()) {
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
    if (mode == MetadataMode::Llm) {
        return get_text_with_metadata(MetadataMode::Llm);
    }
    return get_text_with_metadata(mode);
}

DocNode DocNode::with_score(double score) const {
    DocNode node(*this);
    node.relevance_score_ = score;
    node.has_relevance_score_ = true;
    return node;
}

DocNode DocNode::with_sim_score(double score) const {
    DocNode node(*this);
    node.similarity_score_ = score;
    node.has_similarity_score_ = true;
    return node;
}

bool DocNode::has_relevance_score() const {
    return has_relevance_score_;
}

bool DocNode::has_similarity_score() const {
    return has_similarity_score_;
}

double DocNode::relevance_score() const {
    return relevance_score_;
}

double DocNode::similarity_score() const {
    return similarity_score_;
}

void DocNode::invalidate_content_hash() {
    content_hash_dirty_ = true;
}

QADocNode::QADocNode(const std::string& query, const std::string& answer)
    : DocNode(query), answer_(Trim(answer)) {}

QADocNode::QADocNode(const std::string& query, const std::string& answer, const std::string& uid,
                     const std::string& group)
    : DocNode(query), answer_(Trim(answer)) {
    if (!uid.empty()) {
        uid_ = uid;
    }
    group_ = group;
}

const std::string& QADocNode::answer() const {
    return answer_;
}

std::string QADocNode::get_text_with_metadata(MetadataMode mode) const {
    if (mode == MetadataMode::Llm) {
        std::ostringstream oss;
        oss << "query:\n" << text_ << "\nanswer\n" << answer_;
        return oss.str();
    }
    return DocNode::get_text_with_metadata(mode);
}

ImageDocNode::ImageDocNode(const std::string& image_path)
    : DocNode(image_path), image_path_(Trim(image_path)), modality_("image") {
    set_text(image_path_);
}

ImageDocNode::ImageDocNode(const std::string& image_path, const std::string& uid, const std::string& group)
    : DocNode(image_path), image_path_(Trim(image_path)), modality_("image") {
    if (!uid.empty()) {
        uid_ = uid;
    }
    group_ = group;
    set_text(image_path_);
}

const std::string& ImageDocNode::image_path() const {
    return image_path_;
}

std::string ImageDocNode::get_content(MetadataMode mode) const {
    if (mode == MetadataMode::Embed) {
        std::string file_bytes;
        if (!ReadFileBinary(image_path_, &file_bytes)) {
            return "";
        }
        const std::string mime = ImageMimeType(image_path_);
        const std::string base64 = Base64Encode(file_bytes);
        if (mime.empty()) {
            return base64;
        }
        return "data:" + mime + ";base64," + base64;
    }
    return image_path_;
}

void ImageDocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed) {
    Embedding generated;
    const std::string input = get_content(MetadataMode::Embed);
    for (const auto& item : embed) {
        generated[item.first] = item.second(input, modality_);
    }
    std::lock_guard<std::mutex> lock(embedding_mutex_);
    for (const auto& item : generated) {
        embedding_[item.first] = item.second;
    }
}

std::string ImageDocNode::get_text_with_metadata(MetadataMode mode) const {
    (void)mode;
    return image_path_;
}

} // namespace lazyllm

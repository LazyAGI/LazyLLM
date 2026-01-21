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
    : _uid(GenerateUUID()),
      _group(),
      _text(),
      _content_is_list(false),
      _parent(nullptr),
      _children_loaded(false),
      _content_hash(),
      _content_hash_dirty(true),
      _relevance_score(0.0),
      _has_relevance_score(false),
      _similarity_score(0.0),
      _has_similarity_score(false) {}

DocNode::DocNode(const std::string& text) : DocNode() {
    set_text(text);
}

DocNode::DocNode(const DocNode& other)
    : _uid(other._uid),
      _group(other._group),
      _text(other._text),
      _content_is_list(other._content_is_list),
      _content_list(other._content_list),
      _embedding(other._embedding),
      _metadata(other._metadata),
      _global_metadata(other._global_metadata),
      _excluded_embed_metadata_keys(other._excluded_embed_metadata_keys),
      _excluded_llm_metadata_keys(other._excluded_llm_metadata_keys),
      _parent(other._parent),
      _children(other._children),
      _children_loaded(other._children_loaded),
      _embedding_state(other._embedding_state),
      _content_hash(other._content_hash),
      _content_hash_dirty(other._content_hash_dirty),
      _relevance_score(other._relevance_score),
      _has_relevance_score(other._has_relevance_score),
      _similarity_score(other._similarity_score),
      _has_similarity_score(other._has_similarity_score) {}

DocNode& DocNode::operator=(const DocNode& other) {
    if (this == &other) {
        return *this;
    }
    _uid = other._uid;
    _group = other._group;
    _text = other._text;
    _content_is_list = other._content_is_list;
    _content_list = other._content_list;
    _embedding = other._embedding;
    _metadata = other._metadata;
    _global_metadata = other._global_metadata;
    _excluded_embed_metadata_keys = other._excluded_embed_metadata_keys;
    _excluded_llm_metadata_keys = other._excluded_llm_metadata_keys;
    _parent = other._parent;
    _children = other._children;
    _children_loaded = other._children_loaded;
    _embedding_state = other._embedding_state;
    _content_hash = other._content_hash;
    _content_hash_dirty = other._content_hash_dirty;
    _relevance_score = other._relevance_score;
    _has_relevance_score = other._has_relevance_score;
    _similarity_score = other._similarity_score;
    _has_similarity_score = other._has_similarity_score;
    return *this;
}

const std::string& DocNode::uid() const {
    return _uid;
}

const std::string& DocNode::group() const {
    return _group;
}

void DocNode::set_group(const std::string& group) {
    _group = group;
}

bool DocNode::content_is_list() const {
    return _content_is_list;
}

const std::vector<std::string>& DocNode::content_list() const {
    return _content_list;
}

const std::string& DocNode::content_text() const {
    return _text;
}

void DocNode::set_content(const std::string& text) {
    set_text(text);
}

void DocNode::set_content(const std::vector<std::string>& lines) {
    _content_is_list = true;
    _content_list = lines;
    _text = JoinLines(lines);
    invalidate_content_hash();
}

void DocNode::set_text(const std::string& text) {
    _text = text;
    _content_is_list = false;
    _content_list.clear();
    invalidate_content_hash();
}

const std::string& DocNode::get_text() const {
    return _text;
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

std::string DocNode::content_hash() const {
    if (_content_hash_dirty) {
        _content_hash = Sha256Hex(_text);
        _content_hash_dirty = false;
    }
    return _content_hash;
}

DocNode::Embedding& DocNode::embedding() {
    return _embedding;
}

const DocNode::Embedding& DocNode::embedding() const {
    return _embedding;
}

void DocNode::set_embedding(const Embedding& embed) {
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

void DocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed) {
    Embedding generated;
    const std::string input = get_text_with_metadata(MetadataMode::Embed);
    for (const auto& item : embed) {
        generated[item.first] = item.second(input, "");
    }
    std::lock_guard<std::mutex> lock(_embedding_mutex);
    for (const auto& item : generated) {
        _embedding[item.first] = item.second;
    }
}

void DocNode::set_embedding_value(const std::string& key, const std::vector<float>& value) {
    std::lock_guard<std::mutex> lock(_embedding_mutex);
    _embedding[key] = value;
}

void DocNode::check_embedding_state(const std::string& embed_key) const {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(_embedding_mutex);
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
    if (mode == MetadataMode::None) {
        return "";
    }
    std::set<std::string> keys;
    for (const auto& item : _metadata) {
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
    if (mode == MetadataMode::Llm) {
        return get_text_with_metadata(MetadataMode::Llm);
    }
    return get_text_with_metadata(mode);
}

DocNode DocNode::with_score(double score) const {
    DocNode node(*this);
    node._relevance_score = score;
    node._has_relevance_score = true;
    return node;
}

DocNode DocNode::with_sim_score(double score) const {
    DocNode node(*this);
    node._similarity_score = score;
    node._has_similarity_score = true;
    return node;
}

bool DocNode::has_relevance_score() const {
    return _has_relevance_score;
}

bool DocNode::has_similarity_score() const {
    return _has_similarity_score;
}

double DocNode::relevance_score() const {
    return _relevance_score;
}

double DocNode::similarity_score() const {
    return _similarity_score;
}

void DocNode::invalidate_content_hash() {
    _content_hash_dirty = true;
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
    if (mode == MetadataMode::Llm) {
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
    if (mode == MetadataMode::Embed) {
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

void ImageDocNode::do_embedding(const std::unordered_map<std::string, EmbeddingFn>& embed) {
    Embedding generated;
    const std::string input = get_content(MetadataMode::Embed);
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

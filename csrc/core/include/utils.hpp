#pragma once

#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <iomanip>
#include <set>
#include <variant>
#include <any>

namespace lazyllm {

struct RAGMetadataKeys {
    static constexpr std::string_view KB_ID = "kb_id";
    static constexpr std::string_view DOC_ID = "docid";
    static constexpr std::string_view DOC_PATH = "lazyllm_doc_path";
    static constexpr std::string_view DOC_FILE_NAME = "file_name";
    static constexpr std::string_view DOC_FILE_TYPE = "file_type";
    static constexpr std::string_view DOC_FILE_SIZE = "file_size";
    static constexpr std::string_view DOC_CREATION_DATE = "creation_date";
    static constexpr std::string_view DOC_LAST_MODIFIED_DATE = "last_modified_date";
    static constexpr std::string_view DOC_LAST_ACCESSED_DATE = "last_accessed_date";
};

inline std::string JoinLines(const std::vector<std::string>& lines, char delim = '\n') {
    if (lines.empty()) return {};
    std::string out = lines.front();
    for (size_t i = 1; i < lines.size(); ++i) {
        out += delim;
        out += lines[i];
    }
    return out;
}

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& l, const std::vector<T>& r) {
    std::vector<T> out;
    out.reserve(l.size() + r.size());
    out.insert(out.end(), l.begin(), l.end());
    out.insert(out.end(), r.begin(), r.end());
    return out;
}

template <typename T>
std::set<T> SetUnion(const std::set<T>& l, const std::set<T>& r) {
    std::set<T> out;
    std::set_union(l.begin(), l.end(), r.begin(), r.end(), std::inserter(out, out.begin()));
    return out;
}

template <typename T>
std::set<T> SetDiff(const std::set<T>& l, const std::set<T>& r) {
    std::set<T> out;
    std::set_difference(l.begin(), l.end(), r.begin(), r.end(), std::inserter(out, out.begin()));
    return out;
}

inline std::string to_hex(size_t v) {
    std::ostringstream oss;
    oss << std::hex << v;
    return oss.str();
}

inline std::string NumberToString(double v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

inline std::string GenerateUUID() {
    static const char HEX_CHAR[] = "0123456789abcdef";

    // Single static generator per thread.
    static thread_local std::mt19937 GEN(std::random_device{}());
    static thread_local std::uniform_int_distribution<int> DIST(0, 255);

    std::array<unsigned char, 16> bytes{};
    for (auto& b : bytes) b = static_cast<unsigned char>(DIST(GEN));

    // RFC 4122 UUID v4:
    // - Version: high 4 bits of byte 6 are 0100b.
    // - Variant: high 2 bits of byte 8 are 10b.
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);

    std::string out;
    out.reserve(36);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out.push_back('-');
        out.push_back(HEX_CHAR[(bytes[i] >> 4) & 0x0F]);
        out.push_back(HEX_CHAR[bytes[i] & 0x0F]);
    }
    return out;
}

inline std::string VectorToString(const std::vector<std::string>& values) {
    if (values.empty()) return "[]";
    std::string out = "[";
    for (size_t i = 0; i < values.size() - 1; ++i) {
        out += values[i];
        out += ",";
    }
    out += values.back();
    out += "]";
    return out;
}

inline std::string VectorToString(const std::vector<int>& values) {
    if (values.empty()) return "[]";
    std::string out = "[";
    for (size_t i = 0; i < values.size() - 1; ++i) {
        out += std::to_string(values[i]);
        out += ",";
    }
    out += std::to_string(values.back());
    out += "]";
    return out;
}

inline std::string VectorToString(const std::vector<double>& values) {
    if (values.empty()) return "[]";
    std::string out = "[";
    for (size_t i = 0; i < values.size() - 1; ++i) {
        out += NumberToString(values[i]);
        out += ",";
    }
    out += NumberToString(values.back());
    out += "]";
    return out;
}
using MetadataVType = std::variant<
    std::string, std::vector<std::string>,
    int, std::vector<int>,
    double, std::vector<double>
>;
std::string any_to_string(const MetadataVType& value);

inline bool is_adjacent(const std::string_view& left, const std::string_view& right) {
    return left.data() + left.size() == right.data();
}

struct ChunkView {
    std::string_view view;
    bool is_sentence = false;
    int token_size = 0;
};

struct Chunk {
    std::string text;
    bool is_sentence = false;
    int token_size = 0;

    Chunk& operator+=(const Chunk& r) {
        text += r.text;
        is_sentence = is_sentence && r.is_sentence;
        token_size += r.token_size;
        return *this;
    }
};
} // namespace lazyllm

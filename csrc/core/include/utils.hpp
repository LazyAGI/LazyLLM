#pragma once

#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

namespace lazyllm {

// RAG system metadata keys
constexpr std::string_view RAG_KEY_KB_ID = "kb_id";
constexpr std::string_view RAG_KEY_DOC_ID = "docid";
constexpr std::string_view RAG_KEY_DOC_PATH = "lazyllm_doc_path";
constexpr std::string_view RAG_KEY_DOC_FILE_NAME = "file_name";
constexpr std::string_view RAG_KEY_DOC_FILE_TYPE = "file_type";
constexpr std::string_view RAG_KEY_DOC_FILE_SIZE = "file_size";
constexpr std::string_view RAG_KEY_DOC_CREATION_DATE = "creation_date";
constexpr std::string_view RAG_KEY_DOC_LAST_MODIFIED_DATE = "last_modified_date";
constexpr std::string_view RAG_KEY_DOC_LAST_ACCESSED_DATE = "last_accessed_date";

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

inline std::string GenerateUUID() {
    static const char HEX_CHAR[] = "0123456789abcdef";
    static const int SEGS[] = {8, 4, 4, 4, 12};

    // Single static generator per thread.
    static thread_local std::mt19937 GEN(std::random_device{}());
    static thread_local std::uniform_int_distribution<int> DIST(0, 15);

    std::string out;
    out.reserve(36);
    for (int segLength : SEGS) {
        for (int i = 0; i < segLength; ++i)
            out.push_back(HEX_CHAR[DIST(GEN)]);
        if (segLength < 12)
            out.push_back('-');
    }
    return out;
}

} // namespace lazyllm

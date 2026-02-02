#pragma once

#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

namespace lazyllm {

std::string JoinLines(const std::vector<std::string>& lines) {
    if (lines.empty()) return {};
    std::string out = lines.front();
    for (size_t i = 1; i < lines.size(); ++i) {
        out += '\n';
        out += lines[i];
    }
    return out;
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

} // namespace lazyllm

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <encoding.h>
#include <emdedded_resource_reader.h>
#include <modelparams.h>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#ifdef __linux__
#include <limits.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string_view& view) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids) const = 0;
};

class FallbackByteTokenizer final : public Tokenizer {
public:
    FallbackByteTokenizer() = default;
    ~FallbackByteTokenizer() override = default;

    std::vector<int> encode(const std::string_view& view) const override {
        std::vector<int> token_ids;
        token_ids.reserve(view.size());
        for (unsigned char ch : view) {
            token_ids.push_back(static_cast<int>(ch));
        }
        return token_ids;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        std::string text;
        text.reserve(token_ids.size());
        for (int id : token_ids) {
            text.push_back(static_cast<char>(id & 0xFF));
        }
        return text;
    }
};

inline std::string sanitize_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    const auto* data = reinterpret_cast<const unsigned char*>(input.data());
    const size_t n = input.size();
    size_t i = 0;
    while (i < n) {
        int len = 0;
        unsigned char c = data[i];
        if (c < 0x80) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else {
            output.append("\xEF\xBF\xBD"); // U+FFFD replacement character
            ++i;
            continue;
        }
        if (i + static_cast<size_t>(len) > n) {
            output.append("\xEF\xBF\xBD");
            ++i;
            continue;
        }
        bool valid = true;
        for (int j = 1; j < len; ++j) {
            if ((data[i + j] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            output.append("\xEF\xBF\xBD");
            ++i;
            continue;
        }
        if (len == 2) {
            uint32_t cp = ((c & 0x1F) << 6) | (data[i + 1] & 0x3F);
            if (cp < 0x80) valid = false;
        } else if (len == 3) {
            uint32_t cp = ((c & 0x0F) << 12) | ((data[i + 1] & 0x3F) << 6) | (data[i + 2] & 0x3F);
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) valid = false;
        } else if (len == 4) {
            uint32_t cp = ((c & 0x07) << 18) | ((data[i + 1] & 0x3F) << 12) | ((data[i + 2] & 0x3F) << 6) | (data[i + 3] & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) valid = false;
        }
        if (!valid) {
            output.append("\xEF\xBF\xBD");
            ++i;
            continue;
        }
        output.append(input.substr(i, static_cast<size_t>(len)));
        i += static_cast<size_t>(len);
    }
    return output;
}

class TiktokenTokenizer final : public Tokenizer {
public:
    TiktokenTokenizer() = delete;
    explicit TiktokenTokenizer(LanguageModel model)
        : _reader(resource_name(model)), _encoding(GptEncoding::get_encoding(model, &_reader)) {}

    explicit TiktokenTokenizer(std::string_view encoding_name)
        : TiktokenTokenizer(parse_tiktoken_model(encoding_name)) {}

    ~TiktokenTokenizer() override = default;

    std::vector<int> encode(const std::string_view& view) const override {
        return _encoding->encode(std::string(view)); // TODO refactor to string_view
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        return sanitize_utf8(_encoding->decode(token_ids));
    }

private:
    class LazyllmResourceReader final : public IResourceReader {
    public:
        explicit LazyllmResourceReader(const std::string& resource_name) : resource_name_(resource_name) {}

        std::vector<std::string> readLines() override {
            for (const auto& dir : candidate_dirs()) {
                if (dir.empty()) continue;
                auto path = dir / "tokenizers" / resource_name_;
                std::ifstream file(path);
                if (file.is_open()) {
                    std::string line;
                    std::vector<std::string> lines;
                    while (std::getline(file, line)) lines.push_back(line);
                    return lines;
                }
            }
            throw std::runtime_error("Tokenizer resource '" + resource_name_ + "' not found in any search path.");
        }

    private:
        static std::vector<std::filesystem::path> candidate_dirs() {
            std::vector<std::filesystem::path> dirs;
#ifdef _WIN32
            char* env = nullptr;
            size_t env_len = 0;
            if (_dupenv_s(&env, &env_len, "LAZYLLM_TOKENIZER_PATH") == 0 && env) {
                dirs.emplace_back(env);
                free(env);
            }
#else
            if (const char* env = std::getenv("LAZYLLM_TOKENIZER_PATH")) {
                dirs.emplace_back(env);
            }
#endif
            dirs.emplace_back(get_module_dir());
            dirs.emplace_back(get_exe_parent_path());
            return dirs;
        }

        static std::filesystem::path get_module_dir() {
#if defined(__linux__) || defined(__APPLE__)
            Dl_info info;
            if (dladdr(reinterpret_cast<void*>(&get_module_dir), &info)) {
                return std::filesystem::path(info.dli_fname).parent_path();
            }
#endif
            return {};
        }

        static std::filesystem::path get_exe_parent_path() {
#ifdef _WIN32
            wchar_t result[MAX_PATH] = {0};
            GetModuleFileNameW(nullptr, result, MAX_PATH);
            return std::filesystem::path(result).parent_path();
#elif defined(__APPLE__)
            char result[PATH_MAX];
            uint32_t size = sizeof(result);
            if (_NSGetExecutablePath(result, &size) == 0) {
                return std::filesystem::path(result).parent_path();
            }
            return {};
#else
            char result[PATH_MAX];
            ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
            return std::filesystem::path(std::string(result, count > 0 ? count : 0)).parent_path();
#endif
        }

        std::string resource_name_;
    };

    static std::string resource_name(LanguageModel model) {
        switch (model) {
            case LanguageModel::R50K_BASE: return "r50k_base.tiktoken";
            case LanguageModel::P50K_BASE: return "p50k_base.tiktoken";
            case LanguageModel::P50K_EDIT: return "p50k_base.tiktoken";
            case LanguageModel::CL100K_BASE: return "cl100k_base.tiktoken";
            case LanguageModel::O200K_BASE: return "o200k_base.tiktoken";
            case LanguageModel::QWEN_BASE: return "qwen.tiktoken";
        }
        throw std::runtime_error("Unknown language model");
    }

    static bool has_prefix(std::string_view value, std::string_view prefix) {
        return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    }

    static LanguageModel parse_tiktoken_model(std::string_view name) {
        if (name.empty()) return LanguageModel::R50K_BASE;

        // Model-name aliases used by Python tiktoken.encoding_for_model.
        if (name == "gpt-3.5-turbo" || has_prefix(name, "gpt-3.5-turbo-")) return LanguageModel::CL100K_BASE;
        if (name == "text-embedding-ada-002") return LanguageModel::CL100K_BASE;
        if (name == "text-embedding-3-small" || name == "text-embedding-3-large") return LanguageModel::CL100K_BASE;
        if (name == "gpt-4o" || has_prefix(name, "gpt-4o-")) return LanguageModel::O200K_BASE;
        if (name == "gpt-4.1" || has_prefix(name, "gpt-4.1-")) return LanguageModel::O200K_BASE;
        if (name == "gpt-4.5" || has_prefix(name, "gpt-4.5-")) return LanguageModel::O200K_BASE;
        if (name == "gpt-4" || has_prefix(name, "gpt-4-")) return LanguageModel::CL100K_BASE;
        if (name == "o1" || has_prefix(name, "o1-")) return LanguageModel::O200K_BASE;
        if (name == "o3" || has_prefix(name, "o3-")) return LanguageModel::O200K_BASE;
        if (name == "o4-mini" || has_prefix(name, "o4-mini-")) return LanguageModel::O200K_BASE;

        if (name == "gpt2" || name == "r50k_base" || name == "r50k") return LanguageModel::R50K_BASE;
        if (name == "p50k_base" || name == "p50k") return LanguageModel::P50K_BASE;
        if (name == "p50k_edit") return LanguageModel::P50K_EDIT;
        if (name == "cl100k_base" || name == "cl100k") return LanguageModel::CL100K_BASE;
        if (name == "o200k_base" || name == "o200k") return LanguageModel::O200K_BASE;
        if (name == "qwen_base" || name == "qwen") return LanguageModel::QWEN_BASE;

        throw std::runtime_error(
            "Unknown tiktoken encoding/model name: " + std::string(name) +
            ". Expected one of: gpt2, r50k_base, p50k_base, p50k_edit, cl100k_base, o200k_base, qwen_base." +
            "(Case sensitive)");
    }

private:
    LazyllmResourceReader _reader;
    std::shared_ptr<GptEncoding> _encoding;
};

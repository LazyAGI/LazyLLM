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

class TiktokenTokenizer final : public Tokenizer {
public:
    TiktokenTokenizer() = delete;
    explicit TiktokenTokenizer(LanguageModel model)
        : _encoding(load_encoding(model)) {}

    explicit TiktokenTokenizer(std::string_view encoding_name)
        : TiktokenTokenizer(parse_tiktoken_model(encoding_name)) {}

    ~TiktokenTokenizer() override = default;

    std::vector<int> encode(const std::string_view& view) const override {
        return _encoding->encode(std::string(view)); // TODO refactor to string_view
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        return _encoding->decode(token_ids);
    }

private:
    class FilePathResourceReader final : public IResourceReader {
    public:
        explicit FilePathResourceReader(std::filesystem::path resource_path)
            : resource_path_(std::move(resource_path)) {}

        std::vector<std::string> readLines() override {
            std::ifstream file(resource_path_);
            if (!file.is_open()) {
                throw std::runtime_error("Embedded resource '" + resource_path_.string() + "' not found.");
            }
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(file, line)) lines.push_back(line);
            return lines;
        }

    private:
        std::filesystem::path resource_path_;
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

    static std::shared_ptr<GptEncoding> load_encoding(LanguageModel model) {
        try {
            return GptEncoding::get_encoding(model);
        } catch (const std::exception&) {
            const std::filesystem::path repo_root =
                std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
            const std::string file_name = resource_name(model);
            const std::vector<std::filesystem::path> candidates = {
                repo_root / "build" / "tokenizers" / file_name,
                repo_root / "tokenizers" / file_name,
                std::filesystem::current_path() / "build" / "tokenizers" / file_name,
                std::filesystem::current_path() / "tokenizers" / file_name
            };
            for (const auto& path : candidates) {
                if (!std::filesystem::exists(path)) continue;
                FilePathResourceReader reader(path);
                return GptEncoding::get_encoding(model, &reader);
            }
            throw;
        }
    }

    static bool has_prefix(std::string_view value, std::string_view prefix) {
        return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    }

    static LanguageModel parse_tiktoken_model(std::string_view name) {
        if (name.empty()) return LanguageModel::R50K_BASE;

        // Model-name aliases used by Python tiktoken.encoding_for_model.
        if (name == "gpt-3.5-turbo" || has_prefix(name, "gpt-3.5-turbo-")) return LanguageModel::CL100K_BASE;
        if (name == "gpt-4" || has_prefix(name, "gpt-4-")) return LanguageModel::CL100K_BASE;
        if (name == "text-embedding-ada-002") return LanguageModel::CL100K_BASE;
        if (name == "text-embedding-3-small" || name == "text-embedding-3-large") return LanguageModel::CL100K_BASE;
        if (name == "gpt-4o" || has_prefix(name, "gpt-4o-")) return LanguageModel::O200K_BASE;
        if (name == "gpt-4.1" || has_prefix(name, "gpt-4.1-")) return LanguageModel::O200K_BASE;
        if (name == "gpt-4.5" || has_prefix(name, "gpt-4.5-")) return LanguageModel::O200K_BASE;
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
    std::shared_ptr<GptEncoding> _encoding;
};

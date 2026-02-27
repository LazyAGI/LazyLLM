#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <encoding.h>
#include <modelparams.h>

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string_view& view) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids) const = 0;
};

class TiktokenTokenizer final : public Tokenizer {
public:
    TiktokenTokenizer() = delete;
    explicit TiktokenTokenizer(LanguageModel model)
        : _encoding(GptEncoding::get_encoding(model)) {}

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
    static LanguageModel parse_tiktoken_model(std::string_view name) {
        if (name.empty()) return LanguageModel::R50K_BASE;

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

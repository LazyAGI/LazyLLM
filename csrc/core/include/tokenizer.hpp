#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include <sentencepiece_processor.h>

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string_view& view) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids) const = 0;
};

class SentencePieceTokenizer final : public Tokenizer {
public:
    SentencePieceTokenizer() = delete;
    explicit SentencePieceTokenizer(const std::string& model_path) {
        auto status = _processor.Load(model_path);
        if (!status.ok())
            throw std::runtime_error("Failed to load sentencepiece model: " + model_path);;
    }

    std::vector<int> encode(const std::string_view& view) const override {
        std::vector<int> ids;
        auto status = _processor.Encode(view, &ids);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return ids;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        std::string text;
        auto status = _processor.Decode(token_ids, &text);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return text;
    }

private:
    sentencepiece::SentencePieceProcessor _processor;
};

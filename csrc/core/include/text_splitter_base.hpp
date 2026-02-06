#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <sentencepiece_processor.h>

#include "doc_node.hpp"
#include "node_transform.hpp"

namespace lazyllm {

struct _Split {
    std::string text;
    bool is_sentence = false;
    int token_size = 0;
};

inline std::vector<std::string> split_text_keep_separator(
    const std::string& text,
    const std::string& separator)
{
    if (separator.empty()) return text.empty() ? std::vector<std::string>() : std::vector<std::string>{text};
    if (text.find(separator) == std::string::npos) return {text};

    std::vector<std::string> result;
    size_t start = 0;
    const size_t sep_len = separator.size();
    while (start < text.size()) {
        const size_t idx = text.find(separator, start);
        if (idx == std::string::npos) {
            result.emplace_back(text.substr(start));
            break;
        }
        if (idx == 0) {
            start = sep_len;
            continue;
        }
        result.emplace_back(text.substr(start, idx - start + sep_len));
        start = idx + sep_len;
    }
    return result;
}

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids) const = 0;
};

class SentencePieceTokenizer final : public Tokenizer {
public:
    SentencePieceTokenizer() = default;
    explicit SentencePieceTokenizer(const std::string& model_path) { load(model_path); }

    bool load(const std::string& model_path) {
        auto status = _processor.Load(model_path);
        if (!status.ok()) return false;
        _loaded = true;
        return true;
    }

    std::vector<int> encode(const std::string& text) const override {
        ensure_loaded();
        std::vector<int> ids;
        auto status = _processor.Encode(text, &ids);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return ids;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        ensure_loaded();
        std::string text;
        auto status = _processor.Decode(token_ids, &text);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return text;
    }

private:
    void ensure_loaded() const {
        if (!_loaded) throw std::runtime_error("SentencePiece model not loaded.");
    }

private:
    sentencepiece::SentencePieceProcessor _processor;
    bool _loaded = false;
};

class _TextSplitterBase : public NodeTransform {
public:
    using SplitFn = std::function<std::vector<std::string>(const std::string&)>;

    explicit _TextSplitterBase(
        std::optional<int> chunk_size = std::nullopt,
        std::optional<int> overlap = std::nullopt,
        std::optional<int> num_workers = std::nullopt,
        std::optional<std::string> sentencepiece_model = std::nullopt)
        : NodeTransform(get_param_value("num_workers", num_workers, 0)),
          _chunk_size(get_param_value("chunk_size", chunk_size, 1024)),
          _overlap(get_param_value("overlap", overlap, 200))
    {
        if (_overlap > _chunk_size) {
            throw std::runtime_error(
                "Got a larger chunk overlap than chunk size, should be smaller.");
        }
        if (_chunk_size <= 0 || _overlap < 0)
            throw std::runtime_error("chunk size should > 0 and overlap should >= 0.");

        if (sentencepiece_model.has_value()) {
            from_sentencepiece_model(*sentencepiece_model);
        } else {
            const char* env_model = std::getenv("LAZYLLM_SENTENCEPIECE_MODEL");
            if (env_model && *env_model != '\0') {
                from_sentencepiece_model(std::string(env_model));
            }
        }
    }

    static void set_default(const std::unordered_map<std::string, int>& params) {
        std::lock_guard<std::recursive_mutex> guard(default_params_lock());
        auto& defaults = default_params();
        for (const auto& [key, value] : params) defaults[key] = value;
    }

    static std::unordered_map<std::string, int> get_default() {
        std::lock_guard<std::recursive_mutex> guard(default_params_lock());
        return default_params();
    }

    static std::optional<int> get_default(const std::string& param_name) {
        std::lock_guard<std::recursive_mutex> guard(default_params_lock());
        auto& params = default_params();
        auto it = params.find(param_name);
        if (it == params.end()) return std::nullopt;
        return it->second;
    }

    static void reset_default() {
        std::lock_guard<std::recursive_mutex> guard(default_params_lock());
        default_params().clear();
    }

    _TextSplitterBase& from_sentencepiece_model(const std::string& model_path) {
        auto sp = std::make_shared<SentencePieceTokenizer>();
        if (!sp->load(model_path))
            throw std::runtime_error("Failed to load sentencepiece model: " + model_path);
        _tokenizer = std::move(sp);
        return *this;
    }

    _TextSplitterBase& set_tokenizer(const std::shared_ptr<Tokenizer>& tokenizer) {
        _tokenizer = tokenizer;
        return *this;
    }

    std::vector<std::string> split_text(const std::string& text, int metadata_size) {
        if (text.empty()) return {""};
        const int effective_chunk_size = _chunk_size - metadata_size;
        if (effective_chunk_size <= 0) {
            throw std::runtime_error(
                "Metadata length is longer than chunk size.");
        }
        auto splits = _split(text, effective_chunk_size);
        return _merge(splits, effective_chunk_size);
    }

    TransformResult transform(DocNode* node, const TransformKwargs& /*kwargs*/) override {
        if (node == nullptr) return {};
        auto chunks = split_text(node->get_text(), _get_metadata_size(node));
        TransformResult out;
        out.reserve(chunks.size());
        for (auto& chunk : chunks) out.emplace_back(std::move(chunk));
        return out;
    }

    virtual void set_split_fns(
        const std::vector<SplitFn>& /*split_fns*/,
        const std::optional<std::vector<SplitFn>>& /*sub_split_fns*/ = std::nullopt) {}

    virtual void add_split_fn(const SplitFn& /*split_fn*/, const std::optional<size_t>& /*index*/ = std::nullopt) {}

    virtual void clear_split_fns() {}

protected:
    virtual std::vector<_Split> _split(const std::string& text, int chunk_size) {
        const int token_size = _token_size(text);
        if (token_size <= chunk_size) return {_Split{text, true, token_size}};

        auto [text_splits, is_sentence] = _get_splits_by_fns(text);
        std::vector<_Split> results;
        for (const auto& segment : text_splits) {
            const int seg_token_size = _token_size(segment);
            if (seg_token_size <= chunk_size) {
                results.push_back(_Split{segment, is_sentence, seg_token_size});
            } else {
                auto sub_results = _split(segment, chunk_size);
                results.insert(results.end(), sub_results.begin(), sub_results.end());
            }
        }
        return results;
    }

    virtual std::vector<std::string> _merge(std::vector<_Split> splits, int chunk_size) {
        if (splits.empty()) return {};
        if (splits.size() == 1) return {splits.front().text};

        _Split end_split = splits.back();
        if (end_split.token_size == chunk_size && _overlap > 0) {
            splits.pop_back();
            auto text_tokens = encode(end_split.text);
            const size_t half = text_tokens.size() / 2;
            std::vector<int> p_tokens(text_tokens.begin(), text_tokens.begin() + half);
            std::vector<int> n_tokens(text_tokens.begin() + half, text_tokens.end());
            std::string p_text = decode(p_tokens);
            std::string n_text = decode(n_tokens);
            splits.push_back(_Split{p_text, end_split.is_sentence, _token_size(p_text)});
            splits.push_back(_Split{n_text, end_split.is_sentence, _token_size(n_text)});
            end_split = splits.back();
        }

        std::vector<std::string> result;
        for (int idx = static_cast<int>(splits.size()) - 2; idx >= 0; --idx) {
            _Split start_split = splits[static_cast<size_t>(idx)];
            if (start_split.token_size <= _overlap &&
                end_split.token_size <= chunk_size - _overlap) {
                const bool is_sentence = start_split.is_sentence && end_split.is_sentence;
                const int token_size = start_split.token_size + end_split.token_size;
                end_split = _Split{start_split.text + end_split.text, is_sentence, token_size};
                continue;
            }

            if (end_split.token_size > chunk_size) {
                throw std::runtime_error("split token size is greater than chunk size.");
            }

            const int remaining_space = chunk_size - end_split.token_size;
            const int overlap_len = std::min({_overlap, remaining_space, start_split.token_size});
            if (overlap_len > 0) {
                auto start_tokens = encode(start_split.text);
                std::vector<int> overlap_tokens(
                    start_tokens.end() - overlap_len, start_tokens.end());
                std::string overlap_text = decode(overlap_tokens);
                end_split = _Split{overlap_text + end_split.text, end_split.is_sentence,
                                   end_split.token_size + overlap_len};
            }

            result.insert(result.begin(), end_split.text);
            end_split = start_split;
        }

        result.insert(result.begin(), end_split.text);
        return result;
    }

    virtual std::pair<std::vector<std::string>, bool> _get_splits_by_fns(
        const std::string& text) const
    {
        auto splits = split_text_keep_separator(text, "\n\n\n");
        if (splits.size() > 1) return {splits, true};

        splits = regex_find_all(text, R"([^.!?。？！]+[.!?。？！]?)");
        if (splits.size() > 1) return {splits, true};

        splits = regex_find_all(text, R"([^,.;。？！]+[,.;。？！]?)");
        if (splits.size() > 1) return {splits, false};

        splits = split_text_keep_separator(text, " ");
        if (splits.size() > 1) return {splits, false};

        return {split_to_chars(text), false};
    }

    int _get_metadata_size(const DocNode* node) const {
        return std::max(
            _token_size(node->get_metadata_string(MetadataMode::EMBED)),
            _token_size(node->get_metadata_string(MetadataMode::LLM)));
    }

    int _token_size(const std::string& text) const {
        return static_cast<int>(encode(text).size());
    }

    std::vector<int> encode(const std::string& text) const {
        if (!_tokenizer) throw std::runtime_error("Tokenizer not initialized.");
        return _tokenizer->encode(text);
    }

    std::string decode(const std::vector<int>& token_ids) const {
        if (!_tokenizer) throw std::runtime_error("Tokenizer not initialized.");
        return _tokenizer->decode(token_ids);
    }

    static std::vector<std::string> regex_find_all(
        const std::string& text, const std::string& pattern)
    {
        std::regex re(pattern);
        std::vector<std::string> out;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
             it != std::sregex_iterator(); ++it) {
            out.emplace_back(it->str());
        }
        if (out.empty()) out.emplace_back(text);
        return out;
    }

    static std::vector<std::string> split_to_chars(const std::string& text) {
        std::vector<std::string> out;
        out.reserve(text.size());
        for (char c : text) out.emplace_back(1, c);
        return out;
    }

    static std::unordered_map<std::string, int>& default_params() {
        static std::unordered_map<std::string, int> params;
        return params;
    }

    static std::recursive_mutex& default_params_lock() {
        static std::recursive_mutex lock;
        return lock;
    }

    static int get_param_value(
        const std::string& param_name,
        const std::optional<int>& value,
        int default_value)
    {
        if (value.has_value()) return *value;
        std::lock_guard<std::recursive_mutex> guard(default_params_lock());
        auto& params = default_params();
        auto it = params.find(param_name);
        if (it != params.end()) return it->second;
        return default_value;
    }

protected:
    int _chunk_size = 1024;
    int _overlap = 200;
    std::shared_ptr<Tokenizer> _tokenizer;
};

class _TokenTextSplitter : public _TextSplitterBase {
public:
    explicit _TokenTextSplitter(
        std::optional<int> chunk_size = std::nullopt,
        std::optional<int> overlap = std::nullopt,
        std::optional<int> num_workers = std::nullopt,
        std::optional<std::string> sentencepiece_model = std::nullopt)
        : _TextSplitterBase(chunk_size, overlap, num_workers, sentencepiece_model) {}

protected:
    std::vector<_Split> _split(const std::string& text, int chunk_size) override {
        const int token_size = _token_size(text);
        if (token_size <= chunk_size) return {_Split{text, true, token_size}};

        std::vector<_Split> results;
        auto tokens = encode(text);
        size_t start_idx = 0;
        size_t end_idx = std::min(start_idx + static_cast<size_t>(chunk_size), tokens.size());
        while (start_idx < tokens.size()) {
            std::vector<int> chunk_tokens(tokens.begin() + start_idx, tokens.begin() + end_idx);
            results.push_back(_Split{decode(chunk_tokens), true, static_cast<int>(chunk_tokens.size())});
            if (end_idx >= tokens.size()) break;
            start_idx = std::min(start_idx + static_cast<size_t>(chunk_size - _overlap), tokens.size());
            end_idx = std::min(start_idx + static_cast<size_t>(chunk_size), tokens.size());
        }
        return results;
    }

    std::vector<std::string> _merge(std::vector<_Split> splits, int /*chunk_size*/) override {
        std::vector<std::string> out;
        out.reserve(splits.size());
        for (auto& split : splits) out.emplace_back(std::move(split.text));
        return out;
    }
};

} // namespace lazyllm

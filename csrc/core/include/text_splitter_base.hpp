#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "doc_node.hpp"
#include "tokenizer.hpp"

namespace lazyllm {

class TextSplitterBase {
public:
    TextSplitterBase(int chunk_size, int overlap, const std::string& encoding_name = "gpt2") :
        _chunk_size(chunk_size),
        _overlap(overlap),
        _tokenizer(std::make_shared<TiktokenTokenizer>(encoding_name)) {}

    std::vector<std::string> split_text(const std::string_view& view, int metadata_size) const;
    static std::vector<std::string_view> split_text_while_keeping_separator(
        const std::string_view& text,
        const std::string_view& separator);

protected:
    virtual std::vector<ChunkView> split_recursive(const std::string_view& view, const int chunk_size) const;
    virtual std::vector<std::string> merge_chunks(const std::vector<ChunkView>& splits, int chunk_size) const;

private:
    std::tuple<std::vector<std::string_view>, bool> split_by_functions(const std::string_view& text) const;

    int get_node_metadata_size(const DocNode& node) const {
        return std::max(
            get_token_size(node.get_metadata_string(MetadataMode::EMBED)),
            get_token_size(node.get_metadata_string(MetadataMode::LLM)));
    }

    int get_token_size(const std::string_view& view) const {
        if (view.empty()) return 0;
        return static_cast<int>(_tokenizer->encode(view).size());
    }

protected:
    std::shared_ptr<Tokenizer> _tokenizer = nullptr;
    int _overlap = 0;
    int _chunk_size = 0;
};

} // namespace lazyllm

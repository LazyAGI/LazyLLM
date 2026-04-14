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

#include "tokenizer.hpp"
#include "utils.hpp"

namespace lazyllm {

class TextSplitterBase {
public:
    TextSplitterBase(unsigned chunk_size, unsigned overlap, const std::string& encoding_name = "gpt2") :
        _chunk_size(chunk_size),
        _overlap(overlap),
        _tokenizer(std::make_shared<TiktokenTokenizer>(encoding_name)) {}

    std::vector<std::string> split_text(const std::string_view& view, int metadata_size) const;
    static std::vector<std::string_view> split_text_while_keeping_separator(
        const std::string_view& text,
        const std::string_view& separator);

    unsigned chunk_size() const { return _chunk_size; }
    void set_chunk_size(unsigned value) { _chunk_size = value; }

    unsigned overlap() const { return _overlap; }
    void set_overlap(unsigned value) { _overlap = value; }

protected:
    virtual std::vector<ChunkView> split_recursive(const std::string_view& view, const int chunk_size) const;
    virtual std::vector<std::string> merge_chunks(std::vector<Chunk> splits, int chunk_size) const;

private:
    std::tuple<std::vector<std::string_view>, bool> split_by_functions(const std::string_view& text) const;

    int get_token_size(const std::string_view& view) const {
        if (view.empty()) return 0;
        return static_cast<int>(_tokenizer->encode(view).size());
    }

protected:
    std::shared_ptr<Tokenizer> _tokenizer = nullptr;
    unsigned _overlap = 0;
    unsigned _chunk_size = 0;
};

} // namespace lazyllm

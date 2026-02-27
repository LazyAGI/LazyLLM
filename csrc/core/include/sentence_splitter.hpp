#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "text_splitter_base.hpp"

namespace lazyllm {

class SentenceSplitter : public TextSplitterBase {
public:
    explicit SentenceSplitter(
        std::optional<unsigned> chunk_size,
        std::optional<unsigned> chunk_overlap,
        std::optional<unsigned> worker_num,
        const std::string& encoding_name = "gpt2")
        : TextSplitterBase(chunk_size, chunk_overlap, worker_num, encoding_name) {}

protected:
    std::vector<std::string> merge_chunks(const std::vector<ChunkView>& splits, int chunk_size) const override;

private:
    void close_chunk(
        std::vector<std::string>& chunks,
        std::vector<Chunk>& cur_chunk,
        int& cur_chunk_len,
        bool& is_chunk_new) const;

    static std::string trim_ascii(std::string_view input);
    static std::string join_parts(const std::vector<Chunk>& parts);
};

} // namespace lazyllm

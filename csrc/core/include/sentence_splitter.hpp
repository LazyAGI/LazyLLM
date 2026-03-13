#pragma once

#include <cstddef>
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
};

} // namespace lazyllm

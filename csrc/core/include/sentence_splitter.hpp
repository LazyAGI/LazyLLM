#pragma once

#include <string>
#include <vector>

#include "text_splitter_base.hpp"

namespace lazyllm {

class SentenceSplitter : public TextSplitterBase {
public:
    explicit SentenceSplitter(
        unsigned chunk_size,
        unsigned chunk_overlap,
        const std::string& encoding_name = "gpt2")
        : TextSplitterBase(chunk_size, chunk_overlap, encoding_name) {}

protected:
    std::vector<std::string> merge_chunks(std::vector<Chunk> splits, int chunk_size) const override;
};

} // namespace lazyllm

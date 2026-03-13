#include "sentence_splitter.hpp"

#include <cctype>
#include <stdexcept>

namespace lazyllm {

std::string join_views(
    size_t string_size,
    std::vector<ChunkView>::const_iterator begin,
    const std::vector<ChunkView>::const_iterator& end
) {
    std::string out;
    out.reserve(string_size);
    while(begin != end) {
        out.append(begin->view);
        ++begin;
    }
    return out;
}

std::vector<std::string> SentenceSplitter::merge_chunks(const std::vector<ChunkView>& chunks, int chunk_size) const {
    std::vector<std::string> out;

    auto iLeft = chunks.begin();
    auto iRight = chunks.begin();
    const auto& iEnd = chunks.end();
    int window_token_sum = 0;
    size_t string_size = 0;

    while (iRight != iEnd) {
        if (iRight->token_size > chunk_size)
            throw std::runtime_error("Chunk size is too big.");

        // Grow right edge to the largest window under chunk_size.
        while (iRight != iEnd && window_token_sum + iRight->token_size <= chunk_size) {
            window_token_sum += iRight->token_size;
            string_size += iRight->view.size();
            ++iRight;
        }

        // Merge chunks witin window.
        out.push_back(join_views(string_size, iLeft, iRight));

        // Shrink left edge to select overlap of next merge.
        while (iRight != iEnd && iLeft != iRight && (
            window_token_sum > _overlap || window_token_sum + iRight->token_size > chunk_size
        )) {
            window_token_sum -= iLeft->token_size;
            string_size -= iLeft->view.size();
            ++iLeft;
        }
        // Now window contains only overlap.
    }

    return out;
}

} // namespace lazyllm

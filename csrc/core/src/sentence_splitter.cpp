#include "sentence_splitter.hpp"

#include <cctype>
#include <stdexcept>

namespace {

std::string join_views(
    size_t string_size,
    std::vector<lazyllm::Chunk>::const_iterator begin,
    const std::vector<lazyllm::Chunk>::const_iterator& end
) {
    std::string out;
    out.reserve(string_size);
    while(begin != end) {
        out.append(begin->text);
        ++begin;
    }
    return out;
}

} // namespace

namespace lazyllm {

std::vector<std::string> SentenceSplitter::merge_chunks(std::vector<Chunk> chunks, int chunk_size) const {
    std::vector<std::string> out;

    auto iLeft = chunks.begin();
    auto iRight = chunks.begin();
    auto iEnd = chunks.end();
    int window_token_sum = 0;
    size_t string_size = 0;

    while (iRight != iEnd) {
        if (iRight->token_size > chunk_size)
            throw std::runtime_error("Chunk size is too big.");

        // Grow right edge to the largest window under chunk_size.
        auto iRightPrev = iRight;
        while (iRight != iEnd && window_token_sum + iRight->token_size <= chunk_size) {
            window_token_sum += iRight->token_size;
            string_size += iRight->text.size();
            ++iRight;
        }
        // If no progress was made, the current chunk is too large to fit.
        if (iRight == iRightPrev) {
            throw std::runtime_error("Chunk token_size exceeds chunk_size; cannot make progress.");
        }

        // Merge chunks within window.
        out.push_back(join_views(string_size, iLeft, iRight));

        // Shrink left edge to select overlap of next merge.
        while (iRight != iEnd && iLeft != iRight && (
            window_token_sum > _overlap || window_token_sum + iRight->token_size > chunk_size
        )) {
            window_token_sum -= iLeft->token_size;
            string_size -= iLeft->text.size();
            ++iLeft;
        }
        // Now window contains only overlap.
    }

    // Keep Python behavior: remove leading/trailing whitespace and drop empty chunks.
    std::vector<std::string> normalized;
    normalized.reserve(out.size());
    for (auto& chunk : out) {
        size_t begin = 0;
        while (begin < chunk.size() && std::isspace(static_cast<unsigned char>(chunk[begin]))) ++begin;
        size_t end = chunk.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(chunk[end - 1]))) --end;
        if (end > begin) normalized.emplace_back(chunk.substr(begin, end - begin));
    }

    return normalized;
}

} // namespace lazyllm

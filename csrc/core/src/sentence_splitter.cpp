#include "sentence_splitter.hpp"

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lazyllm {

std::string SentenceSplitter::trim_ascii(std::string_view input) {
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) ++start;

    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;

    return std::string(input.substr(start, end - start));
}

std::string SentenceSplitter::join_parts(const std::vector<Chunk>& parts) {
    size_t total_len = 0;
    for (const auto& part : parts) total_len += part.text.size();

    std::string out;
    out.reserve(total_len);
    for (const auto& part : parts) out += part.text;
    return out;
}

void SentenceSplitter::close_chunk(
    std::vector<std::string>& chunks,
    std::vector<Chunk>& cur_chunk,
    int& cur_chunk_len,
    bool& is_chunk_new) const
{
    chunks.push_back(join_parts(cur_chunk));
    auto last_chunk = std::move(cur_chunk);
    cur_chunk.clear();
    cur_chunk_len = 0;
    is_chunk_new = true;

    int overlap_len = 0;
    for (auto it = last_chunk.rbegin(); it != last_chunk.rend(); ++it) {
        if (overlap_len + it->token_size > _overlap) break;
        cur_chunk.push_back(*it);
        overlap_len += it->token_size;
        cur_chunk_len += it->token_size;
    }
    std::reverse(cur_chunk.begin(), cur_chunk.end());
}

std::vector<std::string> SentenceSplitter::merge_chunks(const std::vector<ChunkView>& splits, int chunk_size) const {
    std::vector<std::string> chunks;
    std::vector<Chunk> cur_chunk;
    int cur_chunk_len = 0;
    bool is_chunk_new = true;

    size_t i = 0;
    while (i < splits.size()) {
        const auto& cur_split = splits[i];
        if (cur_split.token_size > chunk_size) {
            throw std::runtime_error("Single token exceeded chunk size");
        }

        if (cur_chunk_len + cur_split.token_size > chunk_size && !is_chunk_new) {
            close_chunk(chunks, cur_chunk, cur_chunk_len, is_chunk_new);
            continue;
        }

        if (cur_split.is_sentence || cur_chunk_len + cur_split.token_size <= chunk_size || is_chunk_new) {
            cur_chunk_len += cur_split.token_size;
            cur_chunk.push_back({std::string(cur_split.view), cur_split.is_sentence, cur_split.token_size});
            ++i;
            is_chunk_new = false;
        } else {
            close_chunk(chunks, cur_chunk, cur_chunk_len, is_chunk_new);
        }
    }

    if (!is_chunk_new) chunks.push_back(join_parts(cur_chunk));

    std::vector<std::string> out;
    out.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        auto stripped = trim_ascii(chunk);
        if (!stripped.empty()) out.push_back(std::move(stripped));
    }
    return out;
}

} // namespace lazyllm

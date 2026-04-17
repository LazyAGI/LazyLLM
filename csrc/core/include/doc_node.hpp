#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "utils.hpp"

namespace lazyllm {

enum class MetadataMode { ALL, EMBED, LLM, NONE };

struct DocNodeCore {
    using Metadata = std::unordered_map<std::string, MetadataVType>;

    std::string _text;
    Metadata _metadata;
    std::string _uid;
    std::set<std::string> _excluded_embed_metadata_keys;
    std::set<std::string> _excluded_llm_metadata_keys;

    explicit DocNodeCore(
        const std::string& text,
        const Metadata& metadata = {},
        const std::string& uid = ""
    ) : _text(text),
        _metadata(metadata),
        _uid(uid.empty() ? GenerateUUID() : uid) {}
    explicit DocNodeCore(const char* text, const Metadata& metadata = {}, const std::string& uid = "")
        : DocNodeCore(std::string(text == nullptr ? "" : text), metadata, uid) {}

    DocNodeCore(const DocNodeCore&) = default;
    DocNodeCore& operator=(const DocNodeCore&) = default;
    DocNodeCore(DocNodeCore&&) = default;
    DocNodeCore& operator=(DocNodeCore&&) = default;
    virtual ~DocNodeCore() = default;

    virtual std::string get_metadata_string(MetadataMode mode = MetadataMode::ALL) const;

};

} // namespace lazyllm

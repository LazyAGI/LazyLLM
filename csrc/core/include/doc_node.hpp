#pragma once

#include <memory>
#include <set>
#include <string>
#include <string_view>
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
    virtual ~DocNodeCore() = default;

    virtual std::string get_metadata_string(MetadataMode mode = MetadataMode::ALL) const {
        if (mode == MetadataMode::NONE) return "";

        std::set<std::string> valid_keys;
        for (const auto& [key, _] : _metadata) valid_keys.insert(key);

        if (mode == MetadataMode::LLM)
            valid_keys = SetDiff(valid_keys, _excluded_llm_metadata_keys);
        else if (mode == MetadataMode::EMBED)
            valid_keys = SetDiff(valid_keys, _excluded_embed_metadata_keys);

        std::vector<std::string> kv_strings;
        for (const std::string& key : valid_keys)
            kv_strings.emplace_back(key + ": " + any_to_string(_metadata.at(key)));

        return JoinLines(kv_strings);
    }

};

} // namespace lazyllm

#include "doc_node.hpp"

namespace lazyllm {

std::string DocNodeCore::get_metadata_string(MetadataMode mode) const {
    if (mode == MetadataMode::NONE) return "";

    std::vector<std::string> kv_strings;
    if (mode == MetadataMode::ALL) {
        kv_strings.reserve(_metadata.size());
        for (const auto& [key, val] : _metadata) {
            kv_strings.emplace_back(key + ": " + metadata_value_to_string(val));
        }
    } else {
        std::set<std::string> valid_keys;
        for (const auto& [key, val] : _metadata) {
            (void)val;
            if (mode == MetadataMode::LLM && _excluded_llm_metadata_keys.count(key)) continue;
            if (mode == MetadataMode::EMBED && _excluded_embed_metadata_keys.count(key)) continue;
            valid_keys.insert(key);
        }
        kv_strings.reserve(valid_keys.size());
        for (const std::string& key : valid_keys)
            kv_strings.emplace_back(key + ": " + metadata_value_to_string(_metadata.at(key)));
    }

    return JoinLines(kv_strings);
}

} // namespace lazyllm

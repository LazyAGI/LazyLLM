#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "tokenizer.hpp"

namespace {

class IdentityTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(const std::string_view& view) const override {
        std::vector<int> out;
        out.reserve(view.size());
        for (unsigned char ch : view) out.push_back(static_cast<int>(ch));
        return out;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        std::string out;
        out.reserve(token_ids.size());
        for (int id : token_ids) out.push_back(static_cast<unsigned char>(id));
        return out;
    }
};

} // namespace

TEST(tokenizer, abstract_interface_via_derived_class) {
    std::unique_ptr<Tokenizer> tokenizer = std::make_unique<IdentityTokenizer>();
    const auto ids = tokenizer->encode("abc");

    EXPECT_EQ(ids, (std::vector<int>{97, 98, 99}));
    EXPECT_EQ(tokenizer->decode(ids), "abc");
}

TEST(tiktoken_tokenizer, round_trip_encoding) {
    std::unique_ptr<TiktokenTokenizer> tokenizer;
    try {
        tokenizer = std::make_unique<TiktokenTokenizer>("gpt2");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Tokenizer files unavailable, skipping: " << e.what();
    }
    const std::string text = "hello tokenizer";

    const auto token_ids = tokenizer->encode(text);
    EXPECT_FALSE(token_ids.empty());
    EXPECT_EQ(tokenizer->decode(token_ids), text);
}

TEST(tiktoken_tokenizer, alias_names_map_to_same_encoding) {
    std::unique_ptr<TiktokenTokenizer> gpt2;
    std::unique_ptr<TiktokenTokenizer> r50k;
    try {
        gpt2 = std::make_unique<TiktokenTokenizer>("gpt2");
        r50k = std::make_unique<TiktokenTokenizer>("r50k_base");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Tokenizer files unavailable, skipping: " << e.what();
    }

    // "gpt2" is an alias for "r50k_base" in tiktoken; they must produce identical encodings.
    EXPECT_EQ(gpt2->encode("same input"), r50k->encode("same input"));
}

TEST(tiktoken_tokenizer, unknown_encoding_throws) {
    EXPECT_THROW((void)TiktokenTokenizer("unknown_model"), std::runtime_error);
}

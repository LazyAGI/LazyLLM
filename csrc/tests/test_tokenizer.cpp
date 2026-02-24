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
        for (int id : token_ids) out.push_back(static_cast<char>(id));
        return out;
    }
};

} // namespace

TEST(Tokenizer, AbstractInterfaceViaDerivedClass) {
    std::unique_ptr<Tokenizer> tokenizer = std::make_unique<IdentityTokenizer>();
    const auto ids = tokenizer->encode("abc");

    EXPECT_EQ(ids, (std::vector<int>{97, 98, 99}));
    EXPECT_EQ(tokenizer->decode(ids), "abc");
}

TEST(TiktokenTokenizer, RoundTripEncoding) {
    TiktokenTokenizer tokenizer("gpt2");
    const std::string text = "hello tokenizer";

    const auto token_ids = tokenizer.encode(text);
    EXPECT_FALSE(token_ids.empty());
    EXPECT_EQ(tokenizer.decode(token_ids), text);
}

TEST(TiktokenTokenizer, AliasNamesMapToSameEncoding) {
    TiktokenTokenizer gpt2("gpt2");
    TiktokenTokenizer r50k("r50k_base");

    EXPECT_EQ(gpt2.encode("same input"), r50k.encode("same input"));
}

TEST(TiktokenTokenizer, UnknownEncodingThrows) {
    EXPECT_THROW((void)TiktokenTokenizer("unknown_model"), std::runtime_error);
}

#include <gtest/gtest.h>

#include <any>
#include <string>
#include <unordered_map>

#include "adaptor_base.hpp"

namespace {

class MockAdaptor final : public lazyllm::AdaptorBase {
public:
    mutable int call_count = 0;

    std::any call(
        const std::string& func_name,
        const std::unordered_map<std::string, std::any>& args) const override
    {
        ++call_count;
        if (func_name == "sum") {
            return std::any_cast<int>(args.at("left")) + std::any_cast<int>(args.at("right"));
        }
        return func_name;
    }
};

} // namespace

TEST(adaptor_base, derived_call) {
    MockAdaptor adaptor;
    auto result = adaptor.call("echo_me", {});
    EXPECT_EQ(std::any_cast<std::string>(result), "echo_me");
    EXPECT_EQ(adaptor.call_count, 1);

    result = adaptor.call("sum", {{"left", 3}, {"right", 4}});
    EXPECT_EQ(std::any_cast<int>(result), 7);
}

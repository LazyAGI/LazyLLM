#include <gtest/gtest.h>

#include <any>
#include <string>
#include <unordered_map>

#include "adaptor_base.hpp"

namespace {

class EchoAdaptor final : public lazyllm::AdaptorBase {
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

TEST(AdaptorBase, DerivedCallReceivesArgsAndReturnsAny) {
    EchoAdaptor adaptor;
    const auto result = adaptor.call("sum", {{"left", 3}, {"right", 4}});

    EXPECT_EQ(std::any_cast<int>(result), 7);
    EXPECT_EQ(adaptor.call_count, 1);
}

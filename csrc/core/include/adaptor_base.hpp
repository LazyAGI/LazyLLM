#pragma once

#include <string>
#include <unordered_map>
#include <any>

namespace lazyllm {

struct Arg {
    std::string name;
    std::any value;
};

struct AdaptorBase {
    virtual ~AdaptorBase() = default;
    virtual std::any call(
        const std::string& func_name,
        const std::unordered_map<std::string, std::any>& args) const = 0;
};

} // namespace lazyllm

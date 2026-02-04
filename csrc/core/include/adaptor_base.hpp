#pragma once

#include <string>
#include <unordered_map>
#include <any>

#if defined(__GNUC__) || defined(__clang__)
#define LAZYLLM_HIDDEN __attribute__((visibility("hidden")))
#else
#define LAZYLLM_HIDDEN
#endif

namespace lazyllm {

struct Arg {
    std::string name;
    std::any value;
};

class AdaptorBase {
public:
    virtual ~AdaptorBase() = default;
    virtual std::any call(
        const std::string& func_name,
        const std::unordered_map<std::string, std::any>& args) const = 0;
};

} // namespace lazyllm

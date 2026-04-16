#include "utils.hpp"

#include <type_traits>
#include <typeinfo>
#include <stdexcept>

namespace lazyllm {

std::string any_to_string(const MetadataVType& value) {
    if (!value.has_value()) return "None";
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) return v;
        else if constexpr (std::is_same_v<T, int>) return std::to_string(v);
        else if constexpr (std::is_same_v<T, double>) return NumberToString(v);
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) return VectorToString(v);
        else if constexpr (std::is_same_v<T, std::vector<int>>) return VectorToString(v);
        else if constexpr (std::is_same_v<T, std::vector<double>>) return VectorToString(v);
        else if constexpr (std::is_same_v<T, std::unordered_map<std::string, std::string>>) return MapToString(v);
        else throw std::runtime_error(std::string("Unsupported Metadata value type: ") + typeid(T).name());
    }, *value);
}

} // namespace lazyllm

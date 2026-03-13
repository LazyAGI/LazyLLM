#include "utils.hpp"

#include <type_traits>

namespace lazyllm {

namespace {

std::string ScalarToString(const std::string& value) {
    return value;
}

std::string ScalarToString(int value) {
    return std::to_string(value);
}

std::string ScalarToString(double value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <typename T>
std::string VectorToString(const std::vector<T>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        out += ScalarToString(values[i]);
        if (i + 1 < values.size()) out += ",";
    }
    out += "]";
    return out;
}

std::string AnyToString(const std::any& value) {
    if (!value.has_value()) return "";
    const auto& t = value.type();

    if (t == typeid(std::string)) return std::any_cast<std::string>(value);
    if (t == typeid(const char*)) return std::string(std::any_cast<const char*>(value));
    if (t == typeid(char*)) return std::string(std::any_cast<char*>(value));
    if (t == typeid(bool)) return std::any_cast<bool>(value) ? "1" : "0";
    if (t == typeid(int)) return std::to_string(std::any_cast<int>(value));
    if (t == typeid(long)) return std::to_string(std::any_cast<long>(value));
    if (t == typeid(long long)) return std::to_string(std::any_cast<long long>(value));
    if (t == typeid(unsigned int)) return std::to_string(std::any_cast<unsigned int>(value));
    if (t == typeid(unsigned long)) return std::to_string(std::any_cast<unsigned long>(value));
    if (t == typeid(unsigned long long)) return std::to_string(std::any_cast<unsigned long long>(value));
    if (t == typeid(float)) {
        std::ostringstream oss;
        oss << std::any_cast<float>(value);
        return oss.str();
    }
    if (t == typeid(double)) {
        std::ostringstream oss;
        oss << std::any_cast<double>(value);
        return oss.str();
    }
    if (t == typeid(std::vector<std::string>))
        return VectorToString(std::any_cast<const std::vector<std::string>&>(value));
    if (t == typeid(std::vector<int>))
        return VectorToString(std::any_cast<const std::vector<int>&>(value));
    if (t == typeid(std::vector<double>))
        return VectorToString(std::any_cast<const std::vector<double>&>(value));
    return "<unsupported>";
}

} // namespace

std::string any_to_string(const MetadataVType& value) {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) return ScalarToString(v);
        if constexpr (std::is_same_v<T, int>) return ScalarToString(v);
        if constexpr (std::is_same_v<T, double>) return ScalarToString(v);
        if constexpr (std::is_same_v<T, std::vector<std::string>>) return VectorToString(v);
        if constexpr (std::is_same_v<T, std::vector<int>>) return VectorToString(v);
        if constexpr (std::is_same_v<T, std::vector<double>>) return VectorToString(v);
        if constexpr (std::is_same_v<T, std::any>) return AnyToString(v);
        return std::string("<unsupported>");
    }, value);
}

} // namespace lazyllm

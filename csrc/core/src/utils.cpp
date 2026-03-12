#include "utils.hpp"

namespace lazyllm {

std::string any_to_string(const std::any& value) {
    const auto& t = value.type();
    if (t == typeid(std::string)) return std::any_cast<std::string>(value);
    if (t == typeid(const char*)) return std::string(std::any_cast<const char*>(value));
    if (t == typeid(char*)) return std::string(std::any_cast<char*>(value));
    if (t == typeid(bool)) return std::any_cast<bool>(value) ? "True" : "False";
    if (t == typeid(int)) return std::to_string(std::any_cast<int>(value));
    if (t == typeid(long)) return std::to_string(std::any_cast<long>(value));
    if (t == typeid(long long)) return std::to_string(std::any_cast<long long>(value));
    if (t == typeid(unsigned)) return std::to_string(std::any_cast<unsigned>(value));
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
    return "";
}

}
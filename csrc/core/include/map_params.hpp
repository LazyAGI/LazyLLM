#pragma once

#include <any>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace lazyllm {

class MapParams {
public:
    using MapType = std::unordered_map<std::string, std::any>;

    template <typename T>
    T get_param_value(
        const std::string_view& param_name,
        const std::optional<T>& value,
        const T& default_value) const
    {
        if (value.has_value()) return *value;
        std::lock_guard<std::mutex> guard(_lock);
        auto it = _params.find(std::string(param_name));
        if (it != _params.end()) return std::any_cast<T>(it->second);
        return default_value;
    }

    template <typename T>
    void set_default(const std::string_view& param_name, T value) {
        std::lock_guard<std::mutex> guard(_lock);
        _params[std::string(param_name)] = std::any(value);
    }

    void set_default(const MapType& updates) {
        std::lock_guard<std::mutex> guard(_lock);
        for (const auto& entry : updates) {
            _params[entry.first] = entry.second;
        }
    }

    MapType get_default() const {
        std::lock_guard<std::mutex> guard(_lock);
        return _params;
    }

    template <typename T>
    std::optional<T> get_default(const std::string& param_name) const {
        std::lock_guard<std::mutex> guard(_lock);
        auto it = _params.find(param_name);
        if (it == _params.end()) return std::nullopt;
        return std::any_cast<T>(it->second);
    }

    void reset_default() {
        std::lock_guard<std::mutex> guard(_lock);
        _params.clear();
    }

private:
    mutable std::mutex _lock;
    MapType _params;
};

} // namespace lazyllm

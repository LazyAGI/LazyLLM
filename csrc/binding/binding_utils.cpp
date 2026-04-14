#include "binding_utils.hpp"

#include <type_traits>

namespace lazyllm::pybind_utils {

std::string DumpJson(const py::object& obj) {
    py::object json = py::module_::import("json");
    py::object dumps = json.attr("dumps");
    py::object dumped = dumps(obj, py::arg("ensure_ascii") = false);
    return dumped.cast<std::string>();
}

py::object LoadJson(const std::string& text) {
    py::object json = py::module_::import("json");
    py::object loads = json.attr("loads");
    return loads(py::str(text));
}

bool ExtractStringSequence(const py::object& obj, std::vector<std::string>* out) {
    if (!py::isinstance<py::sequence>(obj) || py::isinstance<py::str>(obj)) return false;
    py::sequence seq = obj.cast<py::sequence>();
    out->clear();
    out->reserve(seq.size());
    for (py::handle item : seq) {
        if (!py::isinstance<py::str>(item)) {
            out->clear();
            return false;
        }
        out->push_back(py::cast<std::string>(item));
    }
    return true;
}

lazyllm::MetadataMode ParseMetadataMode(const py::object& mode) {
    if (mode.is_none()) return lazyllm::MetadataMode::NONE;
    try {
        if (py::hasattr(mode, "name")) {
            const auto name = py::cast<std::string>(mode.attr("name"));
            if (name == "ALL") return lazyllm::MetadataMode::ALL;
            if (name == "EMBED") return lazyllm::MetadataMode::EMBED;
            if (name == "LLM") return lazyllm::MetadataMode::LLM;
            if (name == "NONE") return lazyllm::MetadataMode::NONE;
        }
    } catch (const py::error_already_set&) {
    }
    if (py::isinstance<py::str>(mode)) {
        const auto name = mode.cast<std::string>();
        if (name == "ALL") return lazyllm::MetadataMode::ALL;
        if (name == "EMBED") return lazyllm::MetadataMode::EMBED;
        if (name == "LLM") return lazyllm::MetadataMode::LLM;
        if (name == "NONE") return lazyllm::MetadataMode::NONE;
    }
    if (py::isinstance<py::int_>(mode)) {
        const auto value = mode.cast<int>();
        switch (value) {
        case 0: return lazyllm::MetadataMode::ALL;
        case 1: return lazyllm::MetadataMode::EMBED;
        case 2: return lazyllm::MetadataMode::LLM;
        case 3: return lazyllm::MetadataMode::NONE;
        default: break;
        }
    }
    return lazyllm::MetadataMode::NONE;
}

lazyllm::MetadataVType PyToMetadataValue(const py::handle& value) {
    if (value.is_none()) return std::nullopt;
    if (py::isinstance<py::bool_>(value)) return static_cast<int>(value.cast<bool>());
    if (py::isinstance<py::int_>(value)) return value.cast<int>();
    if (py::isinstance<py::float_>(value)) return value.cast<double>();
    if (py::isinstance<py::str>(value)) return value.cast<std::string>();
    if (py::isinstance<py::dict>(value)) {
        py::dict d = value.cast<py::dict>();
        std::unordered_map<std::string, std::string> out;
        out.reserve(d.size());
        for (auto item : d) {
            const std::string key = py::str(item.first).cast<std::string>();
            const std::string val = py::str(item.second).cast<std::string>();
            out.emplace(key, val);
        }
        return out;
    }

    if (py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value)) {
        py::sequence seq = value.cast<py::sequence>();
        if (seq.empty()) return std::vector<std::string>{};

        bool all_str = true;
        bool all_int = true;
        bool all_numeric = true;

        for (py::handle item : seq) {
            const bool is_str = py::isinstance<py::str>(item);
            const bool is_int = py::isinstance<py::int_>(item) && !py::isinstance<py::bool_>(item);
            const bool is_numeric = is_int || py::isinstance<py::float_>(item) || py::isinstance<py::bool_>(item);
            all_str = all_str && is_str;
            all_int = all_int && is_int;
            all_numeric = all_numeric && is_numeric;
        }

        if (all_str) {
            std::vector<std::string> out;
            out.reserve(seq.size());
            for (py::handle item : seq) out.push_back(py::cast<std::string>(item));
            return out;
        }
        if (all_int) {
            std::vector<int> out;
            out.reserve(seq.size());
            for (py::handle item : seq) out.push_back(py::cast<int>(item));
            return out;
        }
        if (all_numeric) {
            std::vector<double> out;
            out.reserve(seq.size());
            for (py::handle item : seq) out.push_back(py::cast<double>(item));
            return out;
        }
        std::vector<std::string> out;
        out.reserve(seq.size());
        for (py::handle item : seq) out.push_back(py::str(item).cast<std::string>());
        return out;
    }
    return py::str(value).cast<std::string>();
}

py::object MetadataValueToPy(const lazyllm::MetadataVType& value) {
    if (!value.has_value()) return py::none();
    return std::visit([](const auto& v) -> py::object {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) return py::str(v);
        if constexpr (std::is_same_v<T, int>) return py::int_(v);
        if constexpr (std::is_same_v<T, double>) return py::float_(v);
        if constexpr (std::is_same_v<T, std::vector<std::string>>) return py::cast(v);
        if constexpr (std::is_same_v<T, std::vector<int>>) return py::cast(v);
        if constexpr (std::is_same_v<T, std::vector<double>>) return py::cast(v);
        if constexpr (std::is_same_v<T, std::unordered_map<std::string, std::string>>) return py::cast(v);
        return py::none();
    }, *value);
}

} // namespace lazyllm::pybind_utils

#include "binding_utils.hpp"

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

std::any PyToAny(const py::handle& value) {
    if (value.is_none()) return std::string("None");
    if (py::isinstance<py::bool_>(value)) return value.cast<bool>();
    if (py::isinstance<py::int_>(value)) return value.cast<long long>();
    if (py::isinstance<py::float_>(value)) return value.cast<double>();
    if (py::isinstance<py::str>(value)) return value.cast<std::string>();
    return py::str(value).cast<std::string>();
}

py::object AnyToPy(const std::any& value) {
    const auto& t = value.type();
    if (t == typeid(std::string)) return py::str(std::any_cast<std::string>(value));
    if (t == typeid(const char*)) return py::str(std::any_cast<const char*>(value));
    if (t == typeid(char*)) return py::str(std::any_cast<char*>(value));
    if (t == typeid(bool)) return py::bool_(std::any_cast<bool>(value));
    if (t == typeid(int)) return py::int_(std::any_cast<int>(value));
    if (t == typeid(long)) return py::int_(std::any_cast<long>(value));
    if (t == typeid(long long)) return py::int_(std::any_cast<long long>(value));
    if (t == typeid(unsigned int)) return py::int_(std::any_cast<unsigned int>(value));
    if (t == typeid(unsigned long)) return py::int_(std::any_cast<unsigned long>(value));
    if (t == typeid(unsigned long long)) return py::int_(std::any_cast<unsigned long long>(value));
    if (t == typeid(float)) return py::float_(std::any_cast<float>(value));
    if (t == typeid(double)) return py::float_(std::any_cast<double>(value));
    return py::str("<unsupported>");
}

} // namespace lazyllm::pybind_utils

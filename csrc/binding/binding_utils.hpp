#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lazyllm.hpp"
#include "doc_node.hpp"

namespace lazyllm::pybind_utils {

std::string DumpJson(const py::object& obj);
py::object LoadJson(const std::string& text);
bool ExtractStringSequence(const py::object& obj, std::vector<std::string>* out);
lazyllm::MetadataMode ParseMetadataMode(const py::object& mode);
lazyllm::MetadataVType PyToMetadataValue(const py::handle& value);
py::object MetadataValueToPy(const lazyllm::MetadataVType& value);

} // namespace lazyllm::pybind_utils

namespace pybind11::detail {

template <>
struct type_caster<lazyllm::MetadataVType> {
public:
    PYBIND11_TYPE_CASTER(lazyllm::MetadataVType, _("MetadataVType"));

    bool load(handle src, bool) {
        try {
            value = lazyllm::pybind_utils::PyToMetadataValue(src);
            return true;
        } catch (const pybind11::error_already_set&) {
            PyErr_Clear();
            return false;
        } catch (...) {
            return false;
        }
    }

    static handle cast(const lazyllm::MetadataVType& src, return_value_policy, handle) {
        pybind11::object obj = lazyllm::pybind_utils::MetadataValueToPy(src);
        return obj.release();
    }
};

} // namespace pybind11::detail

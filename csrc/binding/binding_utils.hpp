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
std::any PyToAny(const py::handle& value);
py::object AnyToPy(const std::any& value);

} // namespace lazyllm::pybind_utils

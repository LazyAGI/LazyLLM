#pragma once

#include <string>
#include <utility>

#include "lazyllm.hpp"

namespace lazyllm::pybind_utils {

template <typename ClassT>
void RegisterMapAsMutableMapping(ClassT& map_cls) {
    py::object abc = py::module_::import("collections.abc");
    abc.attr("Mapping").attr("register")(map_cls);
    abc.attr("MutableMapping").attr("register")(map_cls);
}

template <typename MapType, typename ClassT, typename ToPyValueFn, typename FromPyValueFn, typename ToPyDictFn>
void BindDictLikeMethods(
    ClassT& map_cls,
    ToPyValueFn to_py_value,
    FromPyValueFn from_py_value,
    ToPyDictFn to_py_dict,
    py::object setdefault_default = py::none()
) {
    map_cls
        .def("get",
            [to_py_value](MapType& self, const std::string& key, py::object default_value) {
                auto it = self.find(key);
                if (it == self.end()) return default_value;
                return to_py_value(it->second);
            },
            py::arg("key"),
            py::arg("default") = py::none())
        .def("pop",
            [to_py_value](MapType& self, const std::string& key) {
                auto it = self.find(key);
                if (it == self.end()) throw py::key_error(py::str(key));
                py::object value = to_py_value(it->second);
                self.erase(it);
                return value;
            },
            py::arg("key"))
        .def("pop",
            [to_py_value](MapType& self, const std::string& key, py::object default_value) {
                auto it = self.find(key);
                if (it == self.end()) return default_value;
                py::object value = to_py_value(it->second);
                self.erase(it);
                return value;
            },
            py::arg("key"),
            py::arg("default"))
        .def("setdefault",
            [to_py_value, from_py_value](MapType& self, const std::string& key, py::object default_value) {
                auto it = self.find(key);
                if (it != self.end()) return to_py_value(it->second);
                auto [inserted, ok] = self.emplace(key, from_py_value(default_value));
                (void)ok;
                return to_py_value(inserted->second);
            },
            py::arg("key"),
            py::arg("default") = setdefault_default)
        .def("copy",
            [to_py_dict](const MapType& self) {
                return to_py_dict(self);
            })
        .def("__copy__",
            [to_py_dict](const MapType& self) {
                return to_py_dict(self);
            })
        .def("__deepcopy__",
            [to_py_dict](const MapType& self, const py::dict& memo) {
                py::object copy = py::module_::import("copy");
                return copy.attr("deepcopy")(to_py_dict(self), memo);
            },
            py::arg("memo"))
        .def("update",
            [from_py_value](MapType& self, py::object other, py::kwargs kwargs) {
                if (!other.is_none()) {
                    py::dict d = py::dict(other);
                    for (auto item : d) {
                        const std::string key = py::cast<std::string>(item.first);
                        py::object value = py::reinterpret_borrow<py::object>(item.second);
                        self[key] = from_py_value(value);
                    }
                }
                for (auto item : kwargs) {
                    const std::string key = py::cast<std::string>(item.first);
                    py::object value = py::reinterpret_borrow<py::object>(item.second);
                    self[key] = from_py_value(value);
                }
            },
            py::arg("other") = py::none())
        .def("__eq__",
            [to_py_dict](const MapType& self, py::object other) {
                py::dict lhs = to_py_dict(self);
                if (py::isinstance<py::dict>(other) || py::hasattr(other, "items")) {
                    return py::bool_(lhs.equal(py::dict(other)));
                }
                return py::bool_(false);
            },
            py::is_operator());
}

} // namespace lazyllm::pybind_utils


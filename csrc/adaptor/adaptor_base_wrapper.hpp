#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "adaptor_base.hpp"


namespace lazyllm {

class AdaptorBaseWrapper : public AdaptorBase {
    pybind11::object _py_obj;
public:
    AdaptorBaseWrapper(const pybind11::object &obj) : _py_obj(obj) {}
    std::any call(
        const std::string& func_name,
        const std::unordered_map<std::string, std::any>& args) const override final
    {
        pybind11::gil_scoped_acquire gil;
        pybind11::object func = pybind11::getattr(_py_obj, func_name.c_str(), pybind11::none());
        return call_impl(func_name, func, args);
    }

    virtual std::any call_impl(
        const std::string& func_name,
        const pybind11::object& func,
        const std::unordered_map<std::string, std::any>& args) const = 0;
};

}
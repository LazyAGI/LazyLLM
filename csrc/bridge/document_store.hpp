#pragma once

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace lazyllm {

class DocumentStore {
public:
    DocumentStore() = delete;
    explicit DocumentStore(pybind11::object &store) : _py_store(store) {}

    bool is_group_active(const std::string& grp) const {
        pybind11::gil_scoped_acquire gil;
        pybind11::object fn = _py_store.attr("is_group_active");
        return fn(grp).cast<bool>();
    }

    pybind11::list get_nodes(
        const std::string& group_names
        const std::string& kb_id,
        const std::vector<std::string>& doc_ids
    ) const {
        pybind11::gil_scoped_acquire gil;
        pybind11::object fn = _py_store.attr("get_nodes");
        pybind11::object result = fn(
            pybind11::arg("group_name") = group_name,
            pybind11::arg("kb_id") = kb_id,
            pybind11::arg("doc_ids") = doc_ids);
        return result.cast<pybind11::list>();
    }

private:
    pybind11::object _py_store;
};

} // namespace lazyllm

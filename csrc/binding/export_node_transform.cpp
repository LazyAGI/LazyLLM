#include "lazyllm.hpp"

#include "doc_node.hpp"
#include "node_transform.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

lazyllm::NodeTransform::TransformKwargs kwargs_to_any_map(const py::kwargs& kwargs) {
    lazyllm::NodeTransform::TransformKwargs out;
    for (auto item : kwargs) {
        const auto key = py::cast<std::string>(item.first);
        out[key] = py::cast<py::object>(item.second);
    }
    return out;
}

std::vector<lazyllm::DocNode*> cast_documents(py::object documents) {
    std::vector<lazyllm::DocNode*> docs;
    if (py::isinstance<py::sequence>(documents) && !py::isinstance<py::str>(documents)) {
        for (auto item : documents) docs.push_back(py::cast<lazyllm::DocNode*>(item));
    } else {
        docs.push_back(documents.cast<lazyllm::DocNode*>());
    }
    return docs;
}

} // namespace

void exportNodeTransform(py::module& m) {
    py::class_<lazyllm::NodeTransform>(m, "NodeTransform")
        .def(py::init<int>(), py::arg("num_workers") = 0)
        .def(
            "batch_forward",
            [](lazyllm::NodeTransform& self,
               py::object documents,
               const std::string& node_group,
               py::kwargs kwargs) {
                auto docs = cast_documents(documents);
                auto kw = kwargs_to_any_map(kwargs);
                return self.batch_forward(docs, node_group, kw);
            },
            py::arg("documents"),
            py::arg("node_group"),
            py::kw_only(),
            py::return_value_policy::reference
        )
        .def(
            "with_name",
            [](lazyllm::NodeTransform& self, py::object name, bool copy) -> lazyllm::NodeTransform& {
                if (name.is_none()) return self;
                self.with_name(name.cast<std::string>(), copy);
                return self;
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("copy") = true,
            py::return_value_policy::reference
        );
}

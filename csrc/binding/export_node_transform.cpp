#include "lazyllm.hpp"

#include "doc_node.hpp"
#include "node_transform.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

class PyNodeTransform : public lazyllm::NodeTransform {
public:
    using lazyllm::NodeTransform::NodeTransform;

    std::vector<lazyllm::DocNode> transform(const lazyllm::DocNode* document) const override {
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(static_cast<const lazyllm::NodeTransform*>(this), "transform");
        if (!overload) throw std::runtime_error("NodeTransform.transform is not implemented.");

        py::object result = overload(document);
        if (!py::isinstance<py::sequence>(result)) {
            throw std::runtime_error("NodeTransform.transform must return a sequence.");
        }

        std::vector<lazyllm::DocNode> out;
        for (auto item : result) {
            py::object obj = py::reinterpret_borrow<py::object>(item);
            if (obj.is_none()) continue;

            if (py::isinstance<py::str>(obj)) {
                std::string text = obj.cast<std::string>();
                if (text.empty()) continue;
                out.emplace_back(std::move(text));
            } else {
                out.emplace_back(obj.cast<lazyllm::DocNode>());
            }
        }
        return out;
    }
};

} // namespace

void exportNodeTransform(py::module& m) {
    py::class_<lazyllm::NodeTransform, PyNodeTransform>(m, "NodeTransform")
        .def(py::init<int>(), py::arg("num_workers") = 0)
        .def("batch_forward",
            [](lazyllm::NodeTransform& self,
               py::object documents,
               const std::string& node_group,
               py::object /*ref_path*/,
               py::kwargs /*kwargs*/) {
                std::vector<lazyllm::DocNode*> docs;
                if (py::isinstance<py::sequence>(documents)) {
                    for (auto item : documents) docs.push_back(py::cast<lazyllm::DocNode*>(item));
                } else
                    docs.push_back(documents.cast<lazyllm::DocNode*>());
                return self.batch_forward(docs, node_group);
            },
            py::arg("documents"),
            py::arg("node_group"),
            py::arg("ref_path") = py::none(),
            py::return_value_policy::reference
        )
        .def("transform",
            [](const lazyllm::NodeTransform& self, lazyllm::DocNode* document, py::kwargs /*kwargs*/) {
                if (document == nullptr) return std::vector<lazyllm::DocNode>{};
                return self.transform(document);
            },
            py::arg("document")
        )
        .def("__call__",
            [](const lazyllm::NodeTransform& self, lazyllm::DocNode* node, py::kwargs /*kwargs*/) {
                if (node == nullptr) return std::vector<lazyllm::DocNode>{};
                return self(*node);
            },
            py::arg("node")
        )
        .def(
            "with_name",
            [](lazyllm::NodeTransform& self, py::object name, bool copy) -> lazyllm::NodeTransform& {
                (void)copy;
                if (name.is_none()) return self;
                self._name = name.cast<std::string>();
                return self;
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("copy") = true,
            py::return_value_policy::reference
        )
        .def_readwrite("_name", &lazyllm::NodeTransform::_name)
        .def_property_readonly("_number_workers", &lazyllm::NodeTransform::worker_num);
}

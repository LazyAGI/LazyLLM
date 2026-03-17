#include "lazyllm.hpp"

#include "doc_node.hpp"
#include "node_transform.hpp"

#include <Python.h>

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

class PyNodeTransform : public lazyllm::NodeTransform {
public:
    using lazyllm::NodeTransform::NodeTransform;

    std::vector<lazyllm::PDocNode> transform(lazyllm::PDocNode) const override {
        throw std::runtime_error("NodeTransform.transform is not implemented.");
    }
};

py::object GetBaseModule() {
    return py::module_::import("lazyllm.tools.rag.transform.base");
}

py::object GetRuleSetClass() {
    return GetBaseModule().attr("RuleSet");
}

py::object GetDocNodeClass() {
    return py::module_::import("lazyllm.tools.rag.doc_node").attr("DocNode");
}

py::object GetRichDocNodeClass() {
    return py::module_::import("lazyllm.tools.rag.doc_node").attr("RichDocNode");
}

bool IsDocNode(const py::object& obj) {
    return py::isinstance(obj, GetDocNodeClass());
}

bool IsRichDocNode(const py::object& obj) {
    return py::isinstance(obj, GetRichDocNodeClass());
}

struct TransformRuntimeState {
    py::object rules = py::none();
    py::object on_match = py::none();
    py::object on_miss = py::none();
};

std::unordered_map<const lazyllm::NodeTransform*, TransformRuntimeState>& GetTransformStates() {
    // Intentionally leaked to avoid py::object teardown during Python finalization.
    static auto* states = new std::unordered_map<const lazyllm::NodeTransform*, TransformRuntimeState>();
    return *states;
}

TransformRuntimeState& GetTransformState(const py::object& self) {
    auto* ptr = self.cast<lazyllm::NodeTransform*>();
    auto& states = GetTransformStates();
    auto it = states.find(ptr);
    if (it == states.end()) {
        TransformRuntimeState state;
        state.rules = GetRuleSetClass()();
        auto inserted = states.emplace(ptr, std::move(state));
        return inserted.first->second;
    }
    return it->second;
}

py::object CallForward(const py::object& self, const py::object& node, const py::dict& kwargs) {
    py::object forward = py::getattr(self, "forward", py::none());
    if (forward.is_none()) {
        throw std::runtime_error("NodeTransform.forward is not implemented.");
    }
    if (kwargs.is_none() || kwargs.empty()) {
        return forward(node);
    }
    return forward(node, **kwargs);
}

void ExtendList(py::list& target, const py::list& src) {
    for (auto item : src) target.append(item);
}

} // namespace

void exportNodeTransform(py::module& m) {
    py::class_<lazyllm::NodeTransform, PyNodeTransform>(m, "NodeTransform", py::dynamic_attr())
        .def(py::init_alias<>())
        .def("forward",
            [](py::object /*self*/, py::object /*node*/, py::kwargs /*kwargs*/) {
                PyErr_SetString(PyExc_NotImplementedError,
                    "Subclasses must implement forward() to process a single DocNode or RichDocNode");
                throw py::error_already_set();
            },
            py::arg("node")
        )
        .def("__call__",
            [](py::object self, py::object node_or_nodes, py::kwargs kwargs) -> py::object {
                py::list results;
                py::object forward_single = self.attr("_forward_single");
                if (py::isinstance<py::list>(node_or_nodes) || py::isinstance<py::tuple>(node_or_nodes)) {
                    for (auto item : node_or_nodes) {
                        py::object node = py::reinterpret_borrow<py::object>(item);
                        if (!IsDocNode(node)) {
                            throw py::type_error(
                                "__call__() expects DocNode objects, got non-DocNode in list.");
                        }
                        py::list out = py::cast<py::list>(forward_single(node, **kwargs));
                        ExtendList(results, out);
                    }
                    return py::object(results);
                }

                if (!IsDocNode(node_or_nodes)) {
                    throw py::type_error("__call__() expects DocNode or RichDocNode.");
                }
                return forward_single(node_or_nodes, **kwargs);
            }
        )
        .def("_forward_single",
            [](py::object self, py::object node, py::kwargs kwargs) -> py::object {
                const bool support_rich = py::bool_(py::getattr(self, "__support_rich__", py::bool_(false)));
                if (IsRichDocNode(node) && !support_rich) {
                    py::list out;
                    for (auto sub : node.attr("nodes")) {
                        py::object sub_node = py::reinterpret_borrow<py::object>(sub);
                        py::list res = py::cast<py::list>(CallForward(self, sub_node, py::dict(kwargs)));
                        ExtendList(out, res);
                    }
                    return py::object(out);
                }
                return CallForward(self, node, py::dict(kwargs));
            }
        )
        .def(
            "with_name",
            [](py::object self, py::object name, bool copy) -> py::object {
                if (name.is_none()) return self;
                if (copy) {
                    try {
                        py::object copier = py::module_::import("copy").attr("copy");
                        py::object new_self = copier(self);
                        new_self.attr("_name") = name;
                        return new_self;
                    } catch (const py::error_already_set&) {
                        // Fallback to in-place mutation if copy fails.
                    }
                }
                self.attr("_name") = name;
                return self;
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("copy") = true,
            py::return_value_policy::reference
        )
        .def_readwrite("_name", &lazyllm::NodeTransform::_name)
        .def_property("_rules",
            [](py::object self) { return GetTransformState(self).rules; },
            [](py::object self, py::object value) {
                GetTransformState(self).rules = value.is_none() ? GetRuleSetClass()() : value;
            }
        )
        .def_property("_on_match",
            [](py::object self) { return GetTransformState(self).on_match; },
            [](py::object self, py::object value) { GetTransformState(self).on_match = value; }
        )
        .def_property("_on_miss",
            [](py::object self) { return GetTransformState(self).on_miss; },
            [](py::object self, py::object value) { GetTransformState(self).on_miss = value; }
        )
        ;

    m.attr("NodeTransform").attr("__support_rich__") = py::bool_(false);
}

#include "lazyllm.hpp"

#include "doc_node.hpp"
#include "node_transform.hpp"

#include <Python.h>

#include <memory>
#include <unordered_map>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

class PyNodeTransform : public lazyllm::NodeTransform {
public:
    using lazyllm::NodeTransform::NodeTransform;

    std::vector<lazyllm::PDocNode> transform(lazyllm::PDocNode node) const override {
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(static_cast<const lazyllm::NodeTransform*>(this), "transform");
        if (!overload) {
            overload = py::get_override(static_cast<const lazyllm::NodeTransform*>(this), "forward");
        }
        if (!overload) throw std::runtime_error("NodeTransform.transform is not implemented.");

        py::object result = overload(node);
        if (!py::isinstance<py::sequence>(result)) {
            throw std::runtime_error("NodeTransform.transform must return a sequence.");
        }

        std::vector<lazyllm::PDocNode> out;
        for (auto item : result) {
            py::object obj = py::reinterpret_borrow<py::object>(item);
            if (obj.is_none()) continue;
            out.emplace_back(obj.cast<lazyllm::PDocNode>());
        }
        return out;
    }
};

py::object GetBaseModule() {
    return py::module_::import("lazyllm.tools.rag.transform.base");
}

py::object GetRuleSetClass() {
    return GetBaseModule().attr("RuleSet");
}

py::object GetContextClass() {
    return GetBaseModule().attr("_Context");
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

py::dict GetNodeChildrenDict(const py::object& node) {
    try {
        return py::dict(node.attr("children"));
    } catch (const py::error_already_set&) {
        try {
            return py::dict(node.attr("_children"));
        } catch (const py::error_already_set&) {
            return py::dict();
        }
    }
}

void SetNodeChildrenDict(const py::object& node, const py::dict& children) {
    try {
        node.attr("children") = children;
    } catch (const py::error_already_set&) {
        try {
            node.attr("_children") = children;
        } catch (const py::error_already_set&) {
        }
    }
}

py::list GetRefNodes(const py::object& node, const py::list& ref_path) {
    py::list current;
    current.append(node);
    for (auto key_obj : ref_path) {
        const std::string key = py::cast<std::string>(key_obj);
        py::list next;
        for (auto n_obj : current) {
            py::object n = py::reinterpret_borrow<py::object>(n_obj);
            py::dict children = GetNodeChildrenDict(n);
            py::object group_nodes = children.attr("get")(key, py::list());
            for (auto child : group_nodes) next.append(child);
        }
        current = next;
    }
    return current;
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
        .def(py::init([](int num_workers,
                         py::object rules,
                         bool /*return_trace*/,
                         py::kwargs /*kwargs*/) {
                auto inst = std::make_unique<PyNodeTransform>(num_workers);
                TransformRuntimeState state;
                state.rules = rules.is_none() ? GetRuleSetClass()() : rules;
                GetTransformStates()[inst.get()] = std::move(state);
                return inst;
            }),
            py::arg("num_workers") = 0,
            py::arg("rules") = py::none(),
            py::arg("return_trace") = false
        )
        .def("batch_forward",
            [](py::object self,
               py::object documents,
               const std::string& node_group,
               py::object ref_path,
               py::kwargs kwargs) {
                py::list docs;
                if (py::isinstance<py::list>(documents) || py::isinstance<py::tuple>(documents)) {
                    for (auto item : documents) docs.append(py::reinterpret_borrow<py::object>(item));
                } else {
                    docs.append(documents);
                }

                py::list all_outputs;
                const bool support_rich = py::bool_(py::getattr(self, "__support_rich__", py::bool_(false)));

                for (auto node_obj : docs) {
                    py::object node = py::reinterpret_borrow<py::object>(node_obj);
                    py::dict children = GetNodeChildrenDict(node);
                    if (children.contains(py::str(node_group))) {
                        continue;
                    }

                    py::list splits;
                    if (!ref_path.is_none()) {
                        py::list ref_nodes = GetRefNodes(node, py::cast<py::list>(ref_path));
                        if (ref_nodes.empty()) continue;

                        py::dict forward_kwargs = py::dict(kwargs);
                        forward_kwargs["ref"] = ref_nodes;
                        if (support_rich) {
                            if (py::len(ref_nodes) == 1) {
                                splits = py::cast<py::list>(CallForward(self, ref_nodes[0], forward_kwargs));
                            } else {
                                py::object rich = GetRichDocNodeClass()(py::arg("nodes") = ref_nodes);
                                splits = py::cast<py::list>(CallForward(self, rich, forward_kwargs));
                            }
                        } else {
                            splits = py::list();
                            for (auto ref_node_obj : ref_nodes) {
                                py::object ref_node = py::reinterpret_borrow<py::object>(ref_node_obj);
                                py::list out = py::cast<py::list>(CallForward(self, ref_node, forward_kwargs));
                                ExtendList(splits, out);
                            }
                        }
                    } else {
                        if (IsRichDocNode(node) && !support_rich) {
                            splits = py::list();
                            for (auto sub : node.attr("nodes")) {
                                py::object sub_node = py::reinterpret_borrow<py::object>(sub);
                                py::list out = py::cast<py::list>(CallForward(self, sub_node, kwargs));
                                ExtendList(splits, out);
                            }
                        } else {
                            splits = py::cast<py::list>(CallForward(self, node, kwargs));
                        }
                    }

                    for (auto s_obj : splits) {
                        py::object s = py::reinterpret_borrow<py::object>(s_obj);
                        try {
                            s.attr("parent") = node;
                        } catch (const py::error_already_set&) {
                        }
                        py::setattr(s, "_group", py::str(node_group));
                    }
                    children[py::str(node_group)] = splits;
                    SetNodeChildrenDict(node, children);
                    ExtendList(all_outputs, splits);
                }

                return all_outputs;
            },
            py::arg("documents"),
            py::arg("node_group"),
            py::arg("ref_path") = py::none(),
            py::return_value_policy::reference
        )
        .def("transform",
            [](py::object self, py::object node, py::kwargs kwargs) -> py::object {
                if (node.is_none()) return py::object(py::list());
                return CallForward(self, node, py::dict(kwargs));
            },
            py::arg("document")
        )
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
        .def("process",
            [](py::object self, py::list nodes, py::object on_match, py::object on_miss) {
                py::object rules = py::getattr(self, "_rules", py::none());
                if (rules.is_none()) rules = GetRuleSetClass()();

                py::object match_handler = on_match;
                py::object miss_handler = on_miss;

                py::object instance_match = py::getattr(self, "_on_match", py::none());
                py::object instance_miss = py::getattr(self, "_on_miss", py::none());
                if (match_handler.is_none()) {
                    match_handler = instance_match.is_none()
                        ? py::getattr(self, "_default_match_handler")
                        : instance_match;
                }
                if (miss_handler.is_none()) {
                    miss_handler = instance_miss.is_none()
                        ? py::getattr(self, "_default_miss_handler")
                        : instance_miss;
                }

                py::object ctx = GetContextClass()(py::arg("total") = py::len(nodes));
                py::list results;
                size_t i = 0;
                for (auto node_obj : nodes) {
                    py::object node = py::reinterpret_borrow<py::object>(node_obj);
                    ctx.attr("current_idx") = i++;
                    py::object match = rules.attr("first")(node);
                    py::object processed = match.is_none()
                        ? miss_handler(node, ctx)
                        : match_handler(node, match, ctx);
                    results.append(processed);
                    ctx.attr("prev_node") = node;
                    ctx.attr("prev_result") = processed;
                }
                return results;
            },
            py::arg("nodes"),
            py::arg("on_match") = py::none(),
            py::arg("on_miss") = py::none()
        )
        .def("_default_match_handler",
            [](py::object /*self*/, py::object /*node*/, py::object matched, py::object /*ctx*/) {
                py::tuple tup = matched.cast<py::tuple>();
                return tup[1];
            }
        )
        .def("_default_miss_handler",
            [](py::object /*self*/, py::object node, py::object /*ctx*/) {
                return node;
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
        .def_property("_number_workers",
            &lazyllm::NodeTransform::worker_num,
            &lazyllm::NodeTransform::set_worker_num
        )
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

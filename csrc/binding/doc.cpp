#include "lazyllm.hpp"
#include <iostream>

namespace py = pybind11;

void addDocStr(py::object obj, std::string docs) {
    PyObject* ptr = obj.ptr();
    if (Py_TYPE(ptr) == &PyCFunction_Type) {
        auto f = reinterpret_cast<PyCFunctionObject*>(ptr);
        f->m_ml->ml_doc = strdup(docs.c_str());
    } else if (Py_TYPE(ptr) == &PyInstanceMethod_Type) {
        auto im = reinterpret_cast<PyInstanceMethodObject*>(ptr);
        if (Py_TYPE(im->func) == &PyCFunction_Type) {
            auto f = reinterpret_cast<PyCFunctionObject*>(im->func);
            f->m_ml->ml_doc = strdup(docs.c_str());
        }
    } else if (Py_TYPE(ptr) == &PyMethod_Type) {
        auto m = reinterpret_cast<PyMethodObject*>(ptr);
        if (Py_TYPE(m->im_func) == &PyCFunction_Type) {
            auto f = reinterpret_cast<PyCFunctionObject*>(m->im_func);
            f->m_ml->ml_doc = strdup(docs.c_str());
        } else if (Py_TYPE(m->im_func) == &PyFunction_Type) {
            auto f = reinterpret_cast<PyFunctionObject*>(m->im_func);
            f->func_doc = PyUnicode_FromString(strdup(docs.c_str()));
        }
    } else {
        std::cout << "Adding docstring failed with unexpected type:" << Py_TYPE(ptr)->tp_name << std::endl;
    }
}

void exportDoc(py::module& m) {
    m.def("add_doc", &addDocStr, "Add docstring to a function or method", py::arg("obj"), py::arg("docs"));
}

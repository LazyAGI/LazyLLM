import lazyllm

cpp_add_doc_code = '''
namespace py = pybind11;
void addDocStr(py::object obj, std::string docs) {
    static std::vector<std::string> allDocs;
    PyObject* ptr = obj.ptr();
    allDocs.push_back(std::move(docs));
    if (Py_TYPE(ptr) == &PyCFunction_Type) {
        auto f = reinterpret_cast<PyCFunctionObject*>(ptr);
        f->m_ml->ml_doc = strdup(allDocs.back().c_str());
        return;
    } else if (Py_TYPE(ptr) == &PyInstanceMethod_Type) {
        auto im = reinterpret_cast<PyInstanceMethodObject*>(ptr);
        if (Py_TYPE(im->func) == &PyCFunction_Type) {
            auto f = reinterpret_cast<PyCFunctionObject*>(im->func);
            f->m_ml->ml_doc = strdup(allDocs.back().c_str());
            return;
        }
    }
    allDocs.pop_back();
}
'''

lazyllm.config.add('language', str, 'English', 'LANGUAGE')


def add_doc(obj_name, docstr, module, append=''):
    obj = module
    for n in obj_name.split('.'):
        obj = getattr(obj, n)
    try:
        if append:
            obj.__doc__ += append + docstr
        else:
            obj.__doc__ = docstr
    except Exception:
        raise NotImplementedError('Cannot add doc for builtin class or method now, will be supported in the feature')


def add_chinese_doc(obj_name, docstr, module=lazyllm):
    if lazyllm.config['language'].upper() == 'CHINESE':
        add_doc(obj_name, docstr, module)

def add_english_doc(obj_name, docstr, module=lazyllm):
    if lazyllm.config['language'].upper() == 'ENGLISH':
        add_doc(obj_name, docstr, module)

def add_example(obj_name, docstr, module=lazyllm):
    docstr = '\n'.join([f'    {d}' for d in docstr.split('\n')])
    if lazyllm.config['language'].upper() == 'CHINESE':
        add_doc(obj_name, docstr, module, '\n示例::\n')
    if lazyllm.config['language'].upper() == 'ENGLISH':
        add_doc(obj_name, docstr, module, '\nExample::\n')

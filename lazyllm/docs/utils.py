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

lazyllm.config.add('language', str, 'CHINESE', 'LANGUAGE')

def add_doc(obj_name, docstr, module, append=''):
    obj = module
    for n in obj_name.split('.'):
        obj = getattr(obj, n)
    try:
        if append:
            if isinstance(docstr, str):
                obj.__doc__ += append + docstr
            else:
                cnt = obj.__doc__.count('.. function::')
                if cnt == len(docstr):
                    docs = obj.__doc__.rsplit('.. function::', cnt - 1)
                elif cnt + 1 == len(docstr):
                    docs = obj.__doc__.rsplit('.. function::', cnt)
                else:
                    raise ValueError(f'function number {cnt}, doc number{len(docstr)}')
                obj.__doc__ = '\n.. function::'.join(
                    [(o + append + a) if a.strip() else o for o, a in zip(docs, docstr)])
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
    if isinstance(docstr, str):
        docstr = '\n'.join([f'    {d}' for d in docstr.split('\n')])
    else:
        docstr = ['\n'.join([f'    {d}' for d in doc.split('\n')]) for doc in docstr]
    if lazyllm.config['language'].upper() == 'CHINESE':
        add_doc(obj_name, docstr, module, '\n\nExample::\n')
    if lazyllm.config['language'].upper() == 'ENGLISH':
        add_doc(obj_name, docstr, module, '\n\nExample::\n')

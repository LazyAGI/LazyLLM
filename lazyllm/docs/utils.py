import lazyllm
import ast

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

all_examples = []

def get_all_examples():   # Examples are not always exported, so process them in case of need.
    result = []
    for example in all_examples:
        if len(example.strip()) == 0: continue
        example_lines = []
        code_lines = example.splitlines()
        for code_line in code_lines:
            if code_line.strip().startswith('>>>') or code_line.strip().startswith('...'):
                example_lines.append(code_line.strip()[4:])
            else:
                if len(code_line.strip()) != 0: example_lines.append("# " + code_line)
        result.append("\n".join(example_lines))
    return result

lazyllm.config.add('language', str, 'ENGLISH', 'LANGUAGE')

def add_doc(obj_name, docstr, module, append=''):
    """Add document for lazyllm functions"""
    obj = module
    for n in obj_name.split('.'):
        if isinstance(obj, type): obj = obj.__dict__[n]
        else: obj = getattr(obj, n)
    if isinstance(obj, (classmethod, lazyllm.DynamicDescriptor)): obj = obj.__func__
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


def _extract_assert_triples(source_code):
    # extract triple from assert equation: (left expression, right value, error message)
    tree = ast.parse(source_code)

    triples = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                left_expr = ast.unparse(test.left).strip()
                right_value = ast.unparse(test.comparators[0]).strip()
                err_message = ast.unparse(node.msg).strip() if node.msg else ""
                triples.append((left_expr, right_value, err_message))
    return triples


def rewrite_lines(doc_lines: list[str]) -> list[str]:
    CODE_STARTS = ">>> "
    CODE_CHCHECK_MSG = "LAZYLLM_CHECK_FAILED"
    new_doc_lines = []
    for doc_line in doc_lines:

        if not doc_line.startswith(CODE_STARTS):
            new_doc_lines.append(doc_line)
            continue
        str_remain = doc_line[len(CODE_STARTS):]
        if not str_remain.strip().startswith("assert"):
            new_doc_lines.append(doc_line)
            continue
        triples = _extract_assert_triples(str_remain)
        if len(triples) != 1:
            new_doc_lines.append(doc_line)
            continue
        assert_expr, assert_val, err_msg = triples[0]
        if ast.literal_eval(err_msg) == CODE_CHCHECK_MSG:
            new_doc_lines += [f"{CODE_STARTS}{assert_expr}", f"{assert_val}"]
        else:
            new_doc_lines.append(doc_line)
    return new_doc_lines


def add_example(obj_name, docstr, module=lazyllm):
    if isinstance(docstr, str):
        docstr = "\n".join([f'    {d}' for d in rewrite_lines(docstr.split('\n'))])
        all_examples.append(docstr)
    else:
        docstr = ["\n".join([f'    {d}' for d in rewrite_lines(doc.split('\n'))]) for doc in docstr]
        all_examples.extend(docstr)

    if lazyllm.config['language'].upper() == 'CHINESE':
        add_doc(obj_name, docstr, module, '\n\nExamples:\n')
    if lazyllm.config['language'].upper() == 'ENGLISH':
        add_doc(obj_name, docstr, module, '\n\nExamples:\n')

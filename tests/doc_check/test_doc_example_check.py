import lazyllm
from pathlib import Path
import pytest
from typing import Union
import re


global_func_names = set()
pattern = re.compile(r'^(add_english_doc\(|add_chinese_doc\(|add_example\()')


def add_chinese_doc(obj_name, docstr, module=lazyllm):
    pass


def add_english_doc(obj_name, docstr, module=lazyllm):
    pass


def add_example(obj_name, docstr: Union[str, list], module=lazyllm):
    func_name = "test_" + obj_name.replace(".", "_")
    while func_name in global_func_names:
        func_name = func_name + "_"
    global_func_names.add(func_name)

    if isinstance(docstr, list):
        lines = [d for doc in docstr for d in doc.split("\n")]
    elif isinstance(docstr, str):
        lines = docstr.split("\n")
    else:
        raise TypeError("Expected str or list, got %s" % type(docstr))
    code_lines = []
    for line in lines:
        if line.startswith(">>> ") or line.startswith("... "):
            code_lines.append(f"    {line[4:]}")
    if len(code_lines) == 0:
        return
    xfail_decorator = "@pytest.mark.xfail"
    func_code = f"{xfail_decorator}\ndef {func_name}():\n" + "\n".join(code_lines)
    lazyllm.LOG.info(f"\nTest example:\n{func_code}")
    exec(func_code, globals())


def process_doc(doc_file):
    with open(doc_file, "r", encoding="utf-8") as f:
        doc_lines = f.readlines()
    st_idx = 0
    for i in range(len(doc_lines)):
        match = pattern.match(doc_lines[i])
        if match:
            st_idx = i
            break
    if st_idx == len(doc_lines):
        return
    doc_part = ''.join(doc_lines[st_idx:])
    exec(doc_part, globals())


# 先用一个运行快的例子试一下
doc_files = Path("lazyllm/docs/").glob("flow.py")
for doc_file in doc_files:
    process_doc(doc_file)


if __name__ == "__main__":
    pytest.main()

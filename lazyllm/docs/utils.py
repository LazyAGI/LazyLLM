import lazyllm



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
        docstr = "\n".join([f'    {d}' for d in docstr.split('\n')])
        all_examples.append(docstr)
    else:
        docstr = ["\n".join([f'    {d}' for d in doc.split('\n')]) for doc in docstr]
        all_examples.extend(docstr)

    if lazyllm.config['language'].upper() == 'CHINESE':
        add_doc(obj_name, docstr, module, '\n\nExample::\n')
    if lazyllm.config['language'].upper() == 'ENGLISH':
        add_doc(obj_name, docstr, module, '\n\nExample::\n')

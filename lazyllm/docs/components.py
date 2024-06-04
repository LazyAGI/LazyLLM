from .utils import add_doc as _add_doc
import functools
import lazyllm


add_doc = functools.partial(_add_doc, module=lazyllm.components)


add_doc('Prompter', '''\
This is doc for Prompter
''')

add_doc('Prompter.is_empty', '''\
This is doc for Prompter.is_empty
''')
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.components)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.components)
add_example = functools.partial(utils.add_example, module=lazyllm.components)

# Prompter
add_chinese_doc('Prompter', '''\
这是Prompter的文档
''')
add_english_doc('Prompter', '''\
This is doc for Prompter
''')
add_example('Prompter', '''\
def test_prompter():
    pass
''')

add_english_doc('Prompter.is_empty', '''\
This is doc for Prompter.is_empty
''')

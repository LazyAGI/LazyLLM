import os

language = os.getenv('LAZYLLM_LANGUAGE', 'ENGLISH')
assert language in ('ENGLISH', 'CHINESE')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mkdocs.template.yml')) as f:
    content = f.read()

doc_dir = 'en' if language == 'ENGLISH' else 'zh'
en_default = 'true' if language == 'ENGLISH' else 'false'
zh_default = 'true' if language == 'CHINESE' else 'false'
content = content.format(doc_dir=doc_dir, en_default=en_default, zh_default=zh_default)

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mkdocs.yml'), 'w+') as f:
    f.write(content)

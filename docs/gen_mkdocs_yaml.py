import os
import yaml

language = os.getenv('LAZYLLM_LANGUAGE', 'ENGLISH').upper()
assert language in ('ENGLISH', 'CHINESE')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mkdocs.template.yml')) as f:
    config = yaml.safe_load(f)

doc_dir = 'en' if language == 'ENGLISH' else 'zh'
config['docs_dir'] = f'docs/{doc_dir}'

nav_file = 'nav_en.yml' if language == 'ENGLISH' else 'nav_zh.yml'
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), nav_file)) as f:
    config['nav'] = yaml.safe_load(f)

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mkdocs.yml'), 'w') as f:
    yaml.dump(config, f, allow_unicode=True, sort_keys=False)

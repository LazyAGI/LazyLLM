from lazyllm import pipeline
from lazyllm.tools.data import build_pre_suffix, process_uppercase, AddSuffix, rich_content


def build_demo_pipeline():
    with pipeline() as ppl:
        ppl.buidd_pre_suffix = build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
        ppl.process_uppercase = process_uppercase(input_key='text')
        ppl.add_suffix = AddSuffix(input_key='text', suffix='!!!', _max_workers=4)
        ppl.rich_content = rich_content(input_key='text')
    return ppl

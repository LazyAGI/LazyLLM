from lazyllm import pipeline
from lazyllm.tools.data import demo1, demo2

def build_demo_pipeline(input_key='text'):
    with pipeline() as ppl:
        ppl.build_pre_suffix = demo1.build_pre_suffix(input_key=input_key, prefix='Hello, ', suffix='!')
        ppl.process_uppercase = demo1.process_uppercase(input_key=input_key)
        ppl.add_suffix = demo2.AddSuffix(input_key=input_key, suffix='!!!', _max_workers=4)
        ppl.rich_content = demo2.rich_content(input_key=input_key, _concurrency_mode='single')
    return ppl

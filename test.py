## OP test
from lazyllm.tools.data import build_pre_suffix, process_uppercase, AddSuffix, rich_content

func1 = build_pre_suffix(prefix='Hello, ', suffix='!')
inputs = [{'content': 'world'}, {'content': 'lazyLLM'}]
res1 = func1(inputs)
print(f'The result is: {res1}')

func2 = process_uppercase()
inputs = [{'content': 'hello'}, {'content': 'world'}]
res2 = func2(inputs)
print(f'The result is: {res2}')

func3 = AddSuffix(suffix='!!!', _max_workers=4)
inputs = [{'content': 'exciting'}, {'content': 'amazing'}]
res3 = func3(inputs)
print(f'The result is: {res3}')

func4 = rich_content()
inputs = [{'content': 'This is a test.'}]
res4 = func4(inputs)
print(f'The result is: {res4}')


## Pipeline test
from lazyllm.tools.data import build_demo_pipeline

ppl = build_demo_pipeline()
data = [{'text': 'lazyLLM'} for _ in range(20)]

res = ppl(data)
print(f'\n\nThe final result is: {res}')

from lazyllm.tools.data import DataOperatorRegistry
# tags show:
print(f'\n\nRegistered Data Operators: {DataOperatorRegistry._tags}')

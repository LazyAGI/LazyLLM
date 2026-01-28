# from lazyllm.tools.data import TokenChunker
# from transformers import AutoTokenizer

# # Use a real tokenizer
# print('Loading tokenizer...')
# tokenizer = AutoTokenizer.from_pretrained('/mnt/lustre/share_data/lazyllm/models/qwen2.5-0.5b-instruct', trust_remote_code=True)

# # Initialize TokenChunker with realistic token limits
# chunker = TokenChunker(
#     tokenizer=tokenizer,
#     input_key='content',
#     max_tokens=100,
#     min_tokens=20
# )

# # Test with realistic text
# test_data = [
#     {
#         'content': '''
#         人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

#         该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

#         可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程进行模拟。
#         ''',
#         'meta_data': {'source': 'AI_introduction.txt', 'category': 'tech'}
#     }
# ]

# result = chunker(test_data)
# print(f'\nProcessed {len(test_data)} document(s) -> {len(result)} chunk(s)')
# for i, chunk in enumerate(result):
#     print(f'\n--- Chunk {i+1} ---')
#     print(f'Content: {chunk["content"][:100]}...')
#     print(f'Metadata: {chunk["meta_data"]}')

import json
from lazyllm.tools.data.operator.filter_op import LanguageFilter

# Initialize LanguageFilter to filter Chinese content
# Using single quotes and following snake_case for consistency
print('Initializing LanguageFilter...')
filter_op = LanguageFilter(
    input_key='content',
    target_language='zh',
    threshold=0.5,
    model_path=None,
    model_cache_dir=None
)

# Test data with different languages
test_data = [
    {
        'content': '人工智能是计算机科学的一个分支。它企图了解智能的实质。',
        'meta_data': {'source': 'chinese.txt'}
    },
    {
        'content': 'Artificial Intelligence (AI) is a branch of computer science.',
        'meta_data': {'source': 'english.txt'}
    },
    {
        'content': '你好，这是一个测试。',
        'meta_data': {'source': 'chinese2.txt'}
    },
    {
        'content': 'Hello, this is a test.',
        'meta_data': {'source': 'english2.txt'}
    }
]

print('\nTesting LanguageFilter...')
print('=' * 60)

# Process data - lazyllm operators typically return filtered list or items
raw_results = filter_op(test_data)

# Ensure results is a list and exclude None values
if raw_results is None:
    results = []
elif isinstance(raw_results, list):
    results = [r for r in raw_results if r is not None]
else:
    results = [raw_results]

# Create a list of content that passed for verification
passed_contents = [r.get('content') for r in results]

for i, data in enumerate(test_data):
    content = data.get('content')
    passed = content in passed_contents
    status = '✓ PASSED' if passed else '✗ FILTERED'

    print(f'\n{status} - Test {i+1}')
    print(f'  Source: {data["meta_data"]["source"]}')
    print(f'  Content: {content[:40]}...')

print('\n' + '=' * 60)
print('Test completed!')
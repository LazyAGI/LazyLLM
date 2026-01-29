from lazyllm.tools.data.operator.filter_op import BlocklistFilter


def demo_blocklist_filter() -> None:
    print('=' * 60)
    print('BlocklistFilter demo - Chinese & English')
    print('=' * 60)

    # Test 1: Chinese blocklist filter with jieba tokenizer
    print('\n--- Test 1: Chinese blocklist filter (with jieba) ---')
    filter_zh = BlocklistFilter(
        input_key='content',
        blocklist=['广告', '垃圾', '违禁'],
        language='zh',
        threshold=0,
        use_tokenizer=True
    )

    test_data_zh = [
        {
            'content': '这是一篇正常的文章，讨论人工智能技术。',
            'meta_data': {'source': 'doc_zh_1'}
        },
        {
            'content': '这是一篇包含广告内容的文章。',
            'meta_data': {'source': 'doc_zh_2'}
        },
        {
            'content': '这篇文章讨论机器学习和深度学习。',
            'meta_data': {'source': 'doc_zh_3'}
        },
        {
            'content': '这是垃圾信息，请勿点击违禁链接。',
            'meta_data': {'source': 'doc_zh_4'}
        }
    ]

    print('\nChinese test data:')
    kept = filter_zh(test_data_zh)
    for doc in kept:
        print(f'{doc["content"]}')

    # Test 2: English blocklist filter with nltk tokenizer
    print('\n--- Test 2: English blocklist filter (with nltk) ---')
    filter_en = BlocklistFilter(
        input_key='content',
        blocklist=['spam', 'ads', 'scam'],
        language='en',
        threshold=0,
        use_tokenizer=True
    )

    test_data_en = [
        {
            'content': 'This is a normal article about artificial intelligence.',
            'meta_data': {'source': 'doc_en_1'}
        },
        {
            'content': 'This article contains spam and ads content.',
            'meta_data': {'source': 'doc_en_2'}
        },
        {
            'content': 'Deep learning is a subfield of machine learning.',
            'meta_data': {'source': 'doc_en_3'}
        },
        {
            'content': 'Beware of this scam website trying to steal your data.',
            'meta_data': {'source': 'doc_en_4'}
        }
    ]

    print('\nEnglish test data:')
    kept = filter_en(test_data_en)
    for doc in kept:
        print(f'{doc["content"]}')

    # Test 3: Without tokenizer (simple split)
    print('\n--- Test 3: Without tokenizer (simple split) ---')
    filter_no_tokenizer = BlocklistFilter(
        input_key='content',
        blocklist=['spam', 'ads'],
        language='en',
        threshold=0,
        use_tokenizer=False
    )

    test_data_simple = [
        {
            'content': 'This is clean content',
            'meta_data': {'source': 'doc_simple_1'}
        },
        {
            'content': 'This has spam in it',
            'meta_data': {'source': 'doc_simple_2'}
        }
    ]

    print('\nSimple split test:')
    kept = filter_no_tokenizer(test_data_simple)
    for doc in kept:
        print(f'{doc["content"]}')

    print('\n' + '=' * 60)
    print('Test completed!')


if __name__ == '__main__':
    demo_blocklist_filter()


# ===== Original test code below =====

# from lazyllm.tools.data.operator.filter_op import MinHashDeduplicateFilter


# def demo_minhash_deduplicate() -> None:
#     print('=' * 60)
#     print('MinHashDeduplicateFilter demo')
#     print('=' * 60)

#     dedup_filter = MinHashDeduplicateFilter(
#         input_key='content',
#         threshold=0.85,
#         num_perm=128,
#         use_n_gram=True,
#         ngram=5
#     )

#     test_data = [
#         {
#             'content': 'Artificial intelligence is a branch of computer science.',
#             'meta_data': {'source': 'doc_1'}
#         },
#         {
#             'content': 'Artificial intelligence is a branch of computer science!',
#             'meta_data': {'source': 'doc_1_near_dup'}
#         },
#         {
#             'content': 'Deep learning is a subfield of machine learning.',
#             'meta_data': {'source': 'doc_2'}
#         }
#     ]

#     kept = dedup_filter(test_data)

#     print('\nKept documents (after near-duplicate filtering):')
#     for doc in kept:
#         print(f'- source={doc["meta_data"]["source"]}, content={doc["content"]}')


# if __name__ == '__main__':
#     demo_minhash_deduplicate()

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

# from lazyllm.tools.data.operator.filter_op import LanguageFilter

# # Test 1: Invalid language code (will trigger warning)
# print('=' * 60)
# print('Test 1: Invalid language code')
# print('=' * 60)
# try:
#     filter_op_invalid = LanguageFilter(
#         input_key='content',
#         target_language='zh',
#         threshold=0.5
#     )
#     print('✓ LanguageFilter initialized (warning triggered)\n')
# except Exception as e:
#     print(f'✗ Initialization failed: {e}\n')

# # Test 2: Valid language code
# print('=' * 60)
# print('Test 2: Valid language code')
# print('=' * 60)
# filter_op = LanguageFilter(
#     input_key='content',
#     target_language=['zho_Hans', 'eng_Latn'],
#     threshold=0.5
# )
# print('✓ LanguageFilter initialized successfully\n')

# # Mixed language test data (Chinese + English)
# test_data = [
#     {
#         'content': '人工智能是计算机科学的一个分支。Artificial Intelligence (AI) is a branch of computer science.',
#         'meta_data': {'source': 'mixed_zh_en.txt'}
#     }
# ]

# print('\nTesting LanguageFilter...')
# print('=' * 60)

# raw_results = filter_op(test_data)

# if raw_results is None:
#     results = []
# elif isinstance(raw_results, list):
#     results = [r for r in raw_results if r is not None]
# else:
#     results = [raw_results]

# passed_contents = [r.get('content') for r in results]

# for i, data in enumerate(test_data):
#     content = data.get('content')
#     passed = content in passed_contents
#     status = '✓ PASSED' if passed else '✗ FILTERED'

#     print(f'\n{status} - Test {i+1}')
#     print(f'  Source: {data["meta_data"]["source"]}')
#     print(f'  Content: {content[:40]}...')

# print('\n' + '=' * 60)
# print('Test completed!')



























# from lazyllm.tools.data.operator.filter_op import WordLengthFilter


# def demo_word_length_filter() -> None:
#     print('=' * 60)
#     print('WordLengthFilter demo')
#     print('=' * 60)

#     filter_en = WordLengthFilter(
#         input_key='content',
#         min_length=3,
#         max_length=6,
#     )

#     test_data_en = [
#         {
#             'content': 'AI is great.',
#             'meta_data': {'source': 'doc_en_1', 'note': 'short words'}
#         },
#         {
#             'content': 'This is a normal article.',
#             'meta_data': {'source': 'doc_en_2', 'note': 'normal length'}
#         },
#         {
#             'content': 'Extraordinarily sophisticated implementation.',
#             'meta_data': {'source': 'doc_en_3', 'note': 'very long words'}
#         }
#     ]

#     print('\nTest data (min=3, max=6 avg word length):')
    
#     print(f'\n✓ KEPT:')
#     kept_en = filter_en(test_data_en)
#     print(f'\n✓ KEPT: {len(kept_en)}/{len(test_data_en)}')
#     for doc in kept_en:
#         print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')
    
#     filtered_en = [doc for doc in test_data_en if doc not in kept_en]
#     print(f'\n✗ FILTERED: {len(filtered_en)}/{len(test_data_en)}')
#     for doc in filtered_en:
#         print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')

#     print('\n' + '=' * 60)
#     print('Summary:')
#     print('- Filters based on average word length = total_chars / word_count')
#     print('=' * 60)


# if __name__ == '__main__':
#     demo_word_length_filter()


# ===== Original test code below =====


# from lazyllm.tools.data.operator.filter_op import NullContentFilter, SymbolRatioFilter


# def demo_null_content_filter() -> None:
#     print('=' * 60)
#     print('NullContentFilter demo')
#     print('=' * 60)

#     filter_op = NullContentFilter(input_key='content')

#     test_data = [
#         {
#             'content': 'This is valid content.',
#             'meta_data': {'source': 'doc_1'}
#         },
#         {
#             'content': None,
#             'meta_data': {'source': 'doc_2'}
#         },
#         {
#             'content': '',
#             'meta_data': {'source': 'doc_3'}
#         },
#         {
#             'content': '   ',
#             'meta_data': {'source': 'doc_4'}
#         },
#         {
#             'content': 'Another valid content!',
#             'meta_data': {'source': 'doc_5'}
#         }
#     ]

#     print('\nTest data:')
#     kept = filter_op(test_data)
    
#     print(f'\n✓ KEPT documents ({len(kept)}/{len(test_data)}):')
#     for doc in kept:
#         print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')
    
#     filtered_count = len(test_data) - len(kept)
#     print(f'\n✗ FILTERED documents ({filtered_count}/{len(test_data)}):')
#     for doc in test_data:
#         if doc not in kept:
#             content = doc['content']
#             if content is None:
#                 content = 'None'
#             elif content == '':
#                 content = '(empty string)'
#             elif content.strip() == '':
#                 content = f'(whitespace only: {repr(content)})'
#             print(f'  - [{doc["meta_data"]["source"]}] {content}')

#     print('\n' + '=' * 60)
#     print('Summary: Filters out None, empty strings, and whitespace-only text')
#     print('=' * 60)


# def demo_symbol_ratio_filter() -> None:
#     print('\n\n' + '=' * 60)
#     print('SymbolRatioFilter demo')
#     print('=' * 60)

#     filter_op = SymbolRatioFilter(
#         input_key='content',
#         max_ratio=0.4,
#         symbols=['#', '...', '…']
#     )

#     test_data = [
#         {
#             'content': 'This is a normal text without many symbols.',
#             'meta_data': {'source': 'doc_1', 'note': 'normal'}
#         },
#         {
#             'content': 'This text... has some... symbols... in it...',
#             'meta_data': {'source': 'doc_2', 'note': 'high ellipsis ratio'}
#         },
#         {
#             'content': '#hashtag #trending #viral #popular #content',
#             'meta_data': {'source': 'doc_3', 'note': 'many hashtags'}
#         },
#         {
#             'content': 'AI technology is advancing rapidly.',
#             'meta_data': {'source': 'doc_4', 'note': 'clean text'}
#         },
#         {
#             'content': '待续... 敬请期待… 更多内容... 即将推出…',
#             'meta_data': {'source': 'doc_5', 'note': 'Chinese with ellipsis'}
#         }
#     ]

#     print('\nTest data (max symbol ratio: 40%):')
#     kept = filter_op(test_data)
    
#     print(f'\n✓ KEPT documents ({len(kept)}/{len(test_data)}):')
#     for doc in kept:
#         preview = doc['content'][:50] + '...' if len(doc['content']) > 50 else doc['content']
#         print(f'  - [{doc["meta_data"]["source"]}] {preview}')
    
#     filtered_count = len(test_data) - len(kept)
#     print(f'\n✗ FILTERED documents ({filtered_count}/{len(test_data)}):')
#     for doc in test_data:
#         if doc not in kept:
#             preview = doc['content'][:50] + '...' if len(doc['content']) > 50 else doc['content']
#             print(f'  - [{doc["meta_data"]["source"]}] {preview}')

#     print('\n' + '=' * 60)
#     print('Summary:')
#     print('- Detects symbols: # / ... / …')
#     print('- Filters texts where (symbol_count / word_count) >= 40%')
#     print('- Used to remove hashtag spam and incomplete content')
#     print('=' * 60)


# if __name__ == '__main__':
#     demo_null_content_filter()
#     demo_symbol_ratio_filter()


# ===== Original test code below =====

# from lazyllm.tools.data.operator.filter_op import EllipsisEndFilter


# def demo_ellipsis_end_filter() -> None:
#     print('=' * 60)
#     print('EllipsisEndFilter demo')
#     print('=' * 60)
# 
#     # Create filter (max 30% ellipsis-ending lines)
#     filter_op = EllipsisEndFilter(
#         input_key='content',
#         max_ratio=0.3
#     )
# 
#     test_data = [
#        {
#             'content': '''这是第一行...
# 这是第二行。
# 这是第三行...
# 这是第四行。''',
#             'meta_data': {'source': 'doc_1', 'ellipsis_ratio': '2/4=50%'}
#         },
#         {
#             'content': '''这是第一行。
# 这是第二行。
# 这是第三行...
# 这是第四行。''',
#             'meta_data': {'source': 'doc_2', 'ellipsis_ratio': '1/4=25%'}
#         },
#         {
#             'content': '''完整的第一行。
# 完整的第二行。
# 完整的第三行。''',
#             'meta_data': {'source': 'doc_3', 'ellipsis_ratio': '0/3=0%'}
#         },
#         {
#             'content': '''待续第一行……
# 待续第二行...
# 待续第三行…''',
#             'meta_data': {'source': 'doc_4', 'ellipsis_ratio': '3/3=100%'}
#         }
#     ]

#     print('\nTest data (filter: max 30% ellipsis-ending lines):')
#     kept = filter_op(test_data)
    
#     print(f'\n✓ KEPT documents ({len(kept)}/{len(test_data)}):')
#     for doc in kept:
#         ratio = doc['meta_data']['ellipsis_ratio']
#         preview = ' '.join(doc['content'].split()[:5]) + '...'
#         print(f'  - [{doc["meta_data"]["source"]}] ratio={ratio}: {preview}')
    
#     filtered_count = len(test_data) - len(kept)
#     print(f'\n✗ FILTERED documents ({filtered_count}/{len(test_data)}):')
#     for doc in test_data:
#         if doc not in kept:
#             ratio = doc['meta_data']['ellipsis_ratio']
#             preview = ' '.join(doc['content'].split()[:5]) + '...'
#             print(f'  - [{doc["meta_data"]["source"]}] ratio={ratio}: {preview}')

#     print('\n' + '=' * 60)
#     print('Summary:')
#     print('- Detects lines ending with: ... / … / ……')
#     print('- Filters texts where ellipsis ratio >= max_ratio (30%)')
#     print('- Used to remove incomplete or low-quality content')
#     print('=' * 60)


# if __name__ == '__main__':
#     demo_ellipsis_end_filter()


# from lazyllm.tools.data.operator.filter_op import SentenceCountFilter


# def demo_sentence_count_filter() -> None:
#     print('=' * 60)
#     print('SentenceCountFilter demo - Chinese & English')
#     print('=' * 60)

#     # Test 1: Chinese mode (default)
#     print('\n--- Test 1: Chinese mode (language="zh") ---')
#     filter_zh = SentenceCountFilter(
#         input_key='content',
#         min_sentences=2,
#         max_sentences=4,
#         language='zh'
#     )

#     test_data_zh = [
#         {
#             'content': '1.原理。',
#             'meta_data': {'source': 'zh_1', 'note': 'edge case with number'}
#         },
#         {
#             'content': '这是第一个句子。这是第二个句子。',
#             'meta_data': {'source': 'zh_2', 'expected': 2}
#         },
#         {
#             'content': '人工智能是什么？它能做什么！这很重要。',
#             'meta_data': {'source': 'zh_3', 'expected': 3}
#         },
#         {
#             'content': '句子一。句子二。句子三。句子四。句子五。',
#             'meta_data': {'source': 'zh_4', 'expected': 5}
#         }
#     ]

#     print('Chinese test data (min=2, max=4 sentences):')
#     kept_zh = filter_zh(test_data_zh)
    
#     print(f'\n✓ KEPT ({len(kept_zh)}/{len(test_data_zh)}):')
#     for doc in kept_zh:
#         print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')
    
#     print(f'\n✗ FILTERED ({len(test_data_zh) - len(kept_zh)}/{len(test_data_zh)}):')
#     for doc in test_data_zh:
#         if doc not in kept_zh:
#             print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')

#     # Test 2: English mode
#     print('\n--- Test 2: English mode (language="en") ---')
#     filter_en = SentenceCountFilter(
#         input_key='content',
#         min_sentences=2,
#         max_sentences=4,
#         language='en'
#     )

#     test_data_en = [
#         {
#             'content': 'Mr. Smith works at Google.',
#             'meta_data': {'source': 'en_1', 'note': 'abbreviation test'}
#         },
#         {
#             'content': 'This is sentence one. This is sentence two.',
#             'meta_data': {'source': 'en_2', 'expected': 2}
#         },
#         {
#             'content': 'What is AI? What can it do! This is important.',
#             'meta_data': {'source': 'en_3', 'expected': 3}
#         },
#         {
#             'content': 'One. Two. Three. Four. Five.',
#             'meta_data': {'source': 'en_4', 'expected': 5}
#         }
#     ]

#     print('English test data (min=2, max=4 sentences):')
#     kept_en = filter_en(test_data_en)
    
#     print(f'\n✓ KEPT ({len(kept_en)}/{len(test_data_en)}):')
#     for doc in kept_en:
#         print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')
    
#     print(f'\n✗ FILTERED ({len(test_data_en) - len(kept_en)}/{len(test_data_en)}):')
#     for doc in test_data_en:
#         if doc not in kept_en:
#             print(f'  - [{doc["meta_data"]["source"]}] {doc["content"]}')

#     print('\n' + '=' * 60)
#     print('Summary:')
#     print('- Chinese mode: Only recognizes Chinese punctuation (。！？)')
#     print('  → Correctly handles "1.原理。" as 1 sentence')
#     print('- English mode: Uses NLTK sent_tokenize')
#     print('  → Correctly handles "Mr. Smith" (abbreviation)')
#     print('=' * 60)


# if __name__ == '__main__':
#     demo_sentence_count_filter()


# ===== Original test code below =====

# from lazyllm.tools.data.operator.filter_op import ColonEndFilter


# def demo_colon_end_filter() -> None:
#     print('=' * 60)
#     print('ColonEndFilter demo - Chinese & English')
#     print('=' * 60)

#     # Create filter
#     filter_op = ColonEndFilter(input_key='content')

#     test_data = [
#         {
#             'content': '请问如何使用：',
#             'meta_data': {'source': 'doc_5', 'status': 'incomplete'}
#         },
#         {
#             'content': '请问如何使用这个功能？',
#             'meta_data': {'source': 'doc_6', 'status': 'complete'}
#         },
#         {
#             'content': 'How to use this feature:',
#             'meta_data': {'source': 'doc_7', 'status': 'incomplete'}
#         },
#         {
#             'content': 'How to use this feature?',
#             'meta_data': {'source': 'doc_8', 'status': 'complete'}
#         }
#     ]

#     print('\nTest data (filter out texts ending with colon):')
#     kept = filter_op(test_data)
    
#     print(f'\n✓ KEPT documents ({len(kept)}/{len(test_data)}):')
#     for doc in kept:
#         print(f'  - {doc["content"]}')
    
#     filtered_count = len(test_data) - len(kept)
#     print(f'\n✗ FILTERED documents ({filtered_count}/{len(test_data)}):')
#     for doc in test_data:
#         if doc not in kept:
#             print(f'  - {doc["content"]}')

#     print('\n' + '=' * 60)
#     print('Test completed!')
#     print('Note: Texts ending with ":" or "：" (Chinese colon) are filtered out.')


# if __name__ == '__main__':
#     demo_colon_end_filter()


# ===== Original test code below =====

# from lazyllm.tools.data.operator.filter_op import WordCountFilter


# def demo_word_count_filter() -> None:
#     print('=' * 60)
#     print('WordCountFilter demo - Chinese & English')
#     print('=' * 60)

#     # Test 1: Chinese text - count characters (default)
#     print('\n--- Test 1: Chinese text (character count, default) ---')
#     filter_zh = WordCountFilter(
#         input_key='content',
#         min_words=10,
#         max_words=100,
#         language='zh'
#     )

#     test_data_zh = [
#         {
#             'content': '这是一篇短文。',
#             'meta_data': {'source': 'doc_zh_1', 'chars': 7}
#         },
#         {
#             'content': '这是一篇正常长度的文章，讨论人工智能技术的发展和应用。',
#             'meta_data': {'source': 'doc_zh_2', 'chars': 27}
#         },
#         {
#             'content': '这是一篇非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的文章，超过了字符数限制。',
#             'meta_data': {'source': 'doc_zh_3', 'chars': 102}
#         }
#     ]

#     print('\nChinese test data (min=10, max=100 chars):')
#     kept = filter_zh(test_data_zh)
#     for doc in kept:
#         chars = len(doc['content'].replace(' ', '').replace('\n', '').replace('\t', ''))
#         print(f'✓ KEPT ({chars} chars): {doc["content"][:30]}...')

#     filtered_count = len(test_data_zh) - len(kept)
#     print(f'\nFiltered out: {filtered_count} documents')

#     # Test 2: English text - count words
#     print('\n--- Test 2: English text (word count) ---')
#     filter_en = WordCountFilter(
#         input_key='content',
#         min_words=5,
#         max_words=20,
#         language='en'
#     )

#     test_data_en = [
#         {
#             'content': 'Short text.',
#             'meta_data': {'source': 'doc_en_1', 'words': 2}
#         },
#         {
#             'content': 'This is a normal length article about artificial intelligence.',
#             'meta_data': {'source': 'doc_en_2', 'words': 10}
#         },
#         {
#             'content': 'This is a very very very very very very very very very very very very very very very very very very very very long article that exceeds the word limit.',
#             'meta_data': {'source': 'doc_en_3', 'words': 31}
#         }
#     ]

#     print('\nEnglish test data (min=5, max=20 words):')
#     kept = filter_en(test_data_en)
#     for doc in kept:
#         words = len(doc['content'].split())
#         print(f'✓ KEPT ({words} words): {doc["content"][:50]}...')

#     filtered_count = len(test_data_en) - len(kept)
#     print(f'\nFiltered out: {filtered_count} documents')

#     # Test 3: Mixed comparison
#     print('\n--- Test 3: Summary ---')
#     print(f'Chinese filter (char count): keeps texts with 10-100 characters')
#     print(f'English filter (word count): keeps texts with 5-20 words')

#     print('\n' + '=' * 60)
#     print('Test completed!')


# if __name__ == '__main__':
#     demo_word_count_filter()


# ===== Original test code below =====

# from lazyllm.tools.data.operator.filter_op import BlocklistFilter


# def demo_blocklist_filter() -> None:
#     print('=' * 60)
#     print('BlocklistFilter demo - Chinese & English')
#     print('=' * 60)

#     # Test 1: Chinese blocklist filter with jieba tokenizer
#     print('\n--- Test 1: Chinese blocklist filter (with jieba) ---')
#     filter_zh = BlocklistFilter(
#         input_key='content',
#         blocklist=['广告', '垃圾', '违禁'],
#         language='zh',
#         threshold=0,
#         use_tokenizer=True
#     )

#     test_data_zh = [
#         {
#             'content': '这是一篇正常的文章，讨论人工智能技术。',
#             'meta_data': {'source': 'doc_zh_1'}
#         },
#         {
#             'content': '这是一篇包含广告内容的文章。',
#             'meta_data': {'source': 'doc_zh_2'}
#         },
#         {
#             'content': '这篇文章讨论机器学习和深度学习。',
#             'meta_data': {'source': 'doc_zh_3'}
#         },
#         {
#             'content': '这是垃圾信息，请勿点击违禁链接。',
#             'meta_data': {'source': 'doc_zh_4'}
#         }
#     ]

#     print('\nChinese test data:')
#     kept = filter_zh(test_data_zh)
#     for doc in kept:
#         print(f'{doc["content"]}')

#     # Test 2: English blocklist filter with nltk tokenizer
#     print('\n--- Test 2: English blocklist filter (with nltk) ---')
#     filter_en = BlocklistFilter(
#         input_key='content',
#         blocklist=['spam', 'ads', 'scam'],
#         language='en',
#         threshold=0,
#         use_tokenizer=True
#     )

#     test_data_en = [
#         {
#             'content': 'This is a normal article about artificial intelligence.',
#             'meta_data': {'source': 'doc_en_1'}
#         },
#         {
#             'content': 'This article contains spam and ads content.',
#             'meta_data': {'source': 'doc_en_2'}
#         },
#         {
#             'content': 'Deep learning is a subfield of machine learning.',
#             'meta_data': {'source': 'doc_en_3'}
#         },
#         {
#             'content': 'Beware of this scam website trying to steal your data.',
#             'meta_data': {'source': 'doc_en_4'}
#         }
#     ]

#     print('\nEnglish test data:')
#     kept = filter_en(test_data_en)
#     for doc in kept:
#         print(f'{doc["content"]}')

#     # Test 3: Without tokenizer (simple split)
#     print('\n--- Test 3: Without tokenizer (simple split) ---')
#     filter_no_tokenizer = BlocklistFilter(
#         input_key='content',
#         blocklist=['spam', 'ads'],
#         language='en',
#         threshold=0,
#         use_tokenizer=False
#     )

#     test_data_simple = [
#         {
#             'content': 'This is clean content',
#             'meta_data': {'source': 'doc_simple_1'}
#         },
#         {
#             'content': 'This has spam in it',
#             'meta_data': {'source': 'doc_simple_2'}
#         }
#     ]

#     print('\nSimple split test:')
#     kept = filter_no_tokenizer(test_data_simple)
#     for doc in kept:
#         print(f'{doc["content"]}')

#     print('\n' + '=' * 60)
#     print('Test completed!')


# if __name__ == '__main__':
#     demo_blocklist_filter()


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



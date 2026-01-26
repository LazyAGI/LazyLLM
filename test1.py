from lazyllm.tools.data import TokenChunker
from transformers import AutoTokenizer

# Use a real tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('/mnt/lustre/share_data/lazyllm/models/qwen2.5-0.5b-instruct', trust_remote_code=True)

# Initialize TokenChunker with realistic token limits
chunker = TokenChunker(
    tokenizer=tokenizer,
    input_key='content',
    max_tokens=100,
    min_tokens=20
)

# Test with realistic text
test_data = [
    {
        'content': '''
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

        可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程进行模拟。
        ''',
        'meta_data': {'source': 'AI_introduction.txt', 'category': 'tech'}
    }
]

result = chunker(test_data)
print(f'\nProcessed {len(test_data)} document(s) -> {len(result)} chunk(s)')
for i, chunk in enumerate(result):
    print(f'\n--- Chunk {i+1} ---')
    print(f'Content: {chunk["content"][:100]}...')
    print(f'Metadata: {chunk["meta_data"]}')

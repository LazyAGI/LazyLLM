from typing import List
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register


# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCExpandChunks(kbc):
    def __init__(self, output_key: str = 'raw_chunk', **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.output_key = output_key

    def forward(
        self,
        data: dict,
        **kwargs,
    ) -> List[dict]:
        chunks = data.get('_chunks', [])

        if not chunks:
            return []

        new_records = []
        for chunk_text in chunks:
            new_row = data.copy()
            new_row[self.output_key] = chunk_text
            new_row.pop('_text_content', None)
            new_row.pop('_chunks', None)
            new_records.append(new_row)

        return new_records

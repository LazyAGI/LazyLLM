import random
import json
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.text2qa import Text2MultiHopQAGeneratorPrompt

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadChunkFile(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = "chunk_path",
        **kwargs
    ) -> dict:
        import os
        chunk_path = data.get(input_key, "")
        
        if not chunk_path or not os.path.exists(chunk_path):
            LOG.warning(f"Invalid chunk path: {chunk_path}")
            return {**data, '_chunks_data': []}

        try:
            if str(chunk_path).endswith(".json"):
                with open(chunk_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
            elif str(chunk_path).endswith(".jsonl"):
                with open(chunk_path, "r", encoding="utf-8") as f:
                    file_data = [json.loads(line) for line in f]
            else:
                LOG.warning(f"Unsupported file format: {chunk_path}")
                return {**data, '_chunks_data': []}

            return {**data, '_chunks_data': file_data, '_chunk_path': chunk_path}

        except Exception as e:
            LOG.error(f"Error loading chunk file {chunk_path}: {e}")
            return {**data, '_chunks_data': []}


class KBCPreprocessText(kbc):
    def __init__(self, min_length: int = 100, max_length: int = 200000, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def forward(
        self,
        data: dict,
        text_field: str = "cleaned_chunk",
        **kwargs
    ) -> dict:
        chunks_data = data.get('_chunks_data', [])
        if not chunks_data:
            return {**data, '_processed_chunks': []}

        processed = []
        for item in chunks_data:
            text = item.get(text_field, "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if self.min_length <= len(text) <= self.max_length:
                processed.append({
                    'text': text,
                    'original_data': item
                })

        return {**data, '_processed_chunks': processed}


class KBCExtractInfoPairs(kbc):
    def __init__(self, lang: str = "en", **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.lang = lang

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        processed_chunks = data.get('_processed_chunks', [])
        if not processed_chunks:
            return {**data, '_info_pairs': []}

        all_info_pairs = []
        for chunk in processed_chunks:
            text = chunk.get('text', '')
            original_data = chunk.get('original_data', {})

            if self.lang == "en":
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            else:
                sentences = [s.strip() for s in text.split('。') if s.strip()]

            # Extract triples of sentences for multi-hop reasoning
            for i in range(len(sentences) - 2):
                if len(sentences[i]) > 10 and len(sentences[i + 1]) > 10:
                    info_pair = {
                        'premise': sentences[i],
                        'intermediate': sentences[i + 1],
                        'conclusion': sentences[i + 2] if i + 2 < len(sentences) else '',
                        'related_contexts': [
                            s for j, s in enumerate(sentences)
                            if j != i and j != i + 1 and len(s) > 10
                        ][:2],
                        'original_data': original_data
                    }
                    all_info_pairs.append(info_pair)

        return {**data, '_info_pairs': all_info_pairs}


class KBCBuildMultiHopPrompt(kbc):
    def __init__(self, lang: str = "en", **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.prompt_template = Text2MultiHopQAGeneratorPrompt(lang=lang)

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        info_pairs = data.get('_info_pairs', [])
        if not info_pairs:
            return {**data, '_prompts_data': []}

        prompts_data = []
        for pair in info_pairs:
            context = f"{pair['premise']}. {pair['intermediate']}. {pair['conclusion']}"
            user_prompt = self.prompt_template.build_prompt(context)
            prompts_data.append({
                'user_prompt': user_prompt,
                'info_pair': pair
            })

        return {**data, '_prompts_data': prompts_data}


class KBCGenerateMultiHopQA(kbc):
    def __init__(self, llm=None, lang: str = "en", **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        
        # Initialize prompt template
        self.prompt_template = Text2MultiHopQAGeneratorPrompt(lang=lang)
        
        # Initialize LLM serve with system prompt and formatter
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError("LLM is not configured")

        prompts_data = data.get('_prompts_data', [])
        if not prompts_data:
            return {**data, '_qa_results': []}

        qa_results = []
        for prompt_data in prompts_data:
            user_prompt = prompt_data.get('user_prompt', '')
            info_pair = prompt_data.get('info_pair', {})

            try:
                # Call LLM (system prompt and formatter already set in __init__)
                response = self._llm_serve(user_prompt)
                
                qa_results.append({
                    'response': response,
                    'info_pair': info_pair
                })
            except Exception as e:
                LOG.warning(f"Failed to generate QA: {e}")
                continue

        return {**data, '_qa_results': qa_results}


class KBCParseQAPairs(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        qa_results = data.get('_qa_results', [])
        if not qa_results:
            return {**data, '_qa_pairs': []}

        all_qa_pairs = []
        for qa_result in qa_results:
            response = qa_result.get('response', '')
            info_pair = qa_result.get('info_pair', {})
            original_data = info_pair.get('original_data', {})

            # Parse response as JSON
            # Note: JsonFormatter in LLM has already parsed the response
            # If response is still a string, it means parsing failed and we skip it
            if isinstance(response, dict):
                # Already parsed by JsonFormatter
                if "question" in response:
                    all_qa_pairs.append({
                        **original_data,
                        'qa_pairs': response
                    })
            elif isinstance(response, list):
                # List of QA pairs
                for item in response:
                    if isinstance(item, dict) and "question" in item:
                        all_qa_pairs.append({
                            **original_data,
                            'qa_pairs': item
                        })
            elif isinstance(response, str):
                # JsonFormatter failed to parse, skip this response
                LOG.warning(f"JsonFormatter failed to parse response, skipping: {response[:100]}...")
                continue

        return {**data, '_qa_pairs': all_qa_pairs}


class KBCSaveEnhancedChunks(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        output_key: str = "enhanced_chunk_path",
        **kwargs
    ) -> dict:
        import os
        qa_pairs = data.get('_qa_pairs', [])
        chunk_path = data.get('_chunk_path', '')
        chunks_data = data.get('_chunks_data', [])

        if not chunk_path:
            result = data.copy()
            result[output_key] = ""
            # Clean intermediate fields
            for key in ['_chunks_data', '_chunk_path', '_processed_chunks', 
                       '_info_pairs', '_prompts_data', '_qa_results', '_qa_pairs']:
                result.pop(key, None)
            return result

        # Merge QA pairs back into original data
        enhanced_data = []
        for item in chunks_data:
            enhanced_item = item.copy()
            # Find matching QA pairs for this chunk
            matching_qa = [qa for qa in qa_pairs if qa.get('cleaned_chunk') == item.get('cleaned_chunk')]
            if matching_qa:
                enhanced_item['qa_pairs'] = matching_qa[0].get('qa_pairs', {})
            enhanced_data.append(enhanced_item)

        try:
            # Write back to file
            with open(chunk_path, "w", encoding="utf-8") as f:
                if str(chunk_path).endswith(".json"):
                    json.dump(enhanced_data, f, ensure_ascii=False, indent=4)
                else:
                    for item in enhanced_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            LOG.info(f"Saved enhanced chunks to {chunk_path}")

            result = data.copy()
            result[output_key] = chunk_path
            # Clean intermediate fields
            for key in ['_chunks_data', '_chunk_path', '_processed_chunks', 
                       '_info_pairs', '_prompts_data', '_qa_results', '_qa_pairs']:
                result.pop(key, None)
            return result

        except Exception as e:
            LOG.error(f"Error saving enhanced chunks: {e}")
            result = data.copy()
            result[output_key] = ""
            # Clean intermediate fields
            for key in ['_chunks_data', '_chunk_path', '_processed_chunks', 
                       '_info_pairs', '_prompts_data', '_qa_results', '_qa_pairs']:
                result.pop(key, None)
            return result

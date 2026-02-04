"""KBC Multi-Hop QA Generator Batch operator"""
import random
import json
from typing import List
from tqdm import tqdm
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.text2qa import Text2MultiHopQAGeneratorPrompt

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCMultiHopQAGeneratorBatch(kbc):
    """
    Processor for generating multi-hop question-answer pairs from text data.
    多跳问答对生成处理器。
    """

    def __init__(
            self,
            llm=None,
            seed: int = 0,
            lang: str = "en",
            prompt_template=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.rng = random.Random(seed)
        self.llm = llm
        self.lang = lang
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = Text2MultiHopQAGeneratorPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "MultiHopQAGenerator 是多跳问答对生成处理器，"
                "支持从文本中自动生成需要多步推理的问题与答案。"
            )
        else:
            return (
                "MultiHopQAGenerator is a processor for generating multi-hop "
                "question-answer pairs from raw text."
            )

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm is None:
            raise ValueError("LLM is not configured")
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results

    def _preprocess_text(self, text: str, min_length: int = 100, max_length: int = 200000) -> str:
        """Preprocess input text"""
        if not isinstance(text, str):
            return ''
        text = text.strip()
        if len(text) < min_length or len(text) > max_length:
            return ''
        return text

    def _extract_info_pairs(self, text: str):
        """Extract information pairs from text"""
        if self.lang == "en":
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        else:
            sentences = [s.strip() for s in text.split('。') if s.strip()]

        info_pairs = []
        for i in range(len(sentences) - 2):
            if len(sentences[i]) > 10 and len(sentences[i + 1]) > 10:
                info_pairs.append({
                    'premise': sentences[i],
                    'intermediate': sentences[i + 1],
                    'conclusion': sentences[i + 2] if i + 2 < len(sentences) else '',
                    'related_contexts': [
                        s for j, s in enumerate(sentences)
                        if j != i and j != i + 1 and len(s) > 10
                    ][:2],
                })
        return info_pairs

    def _generate_qa_pairs(self, info_pairs):
        """Generate QA pairs from info pairs"""
        user_inputs = []
        for pair in info_pairs:
            context = f"{pair['premise']}. {pair['intermediate']}. {pair['conclusion']}"
            user_inputs.append(self.prompt_template.build_prompt(context))

        sys_prompt = self.prompt_template.build_system_prompt()
        responses = self._generate_from_llm(user_inputs, sys_prompt)
        return self._extract_qa_pairs(responses)

    def _extract_qa_pairs(self, responses):
        """Extract QA pairs from LLM responses"""
        qa_pairs = []
        for response in responses:
            try:
                qa_pair = json.loads(response)
                if isinstance(qa_pair, dict) and "question" in qa_pair:
                    qa_pairs.append(qa_pair)
                elif isinstance(qa_pair, list):
                    for item in qa_pair:
                        if isinstance(item, dict) and "question" in item:
                            qa_pairs.append(item)
            except json.JSONDecodeError:
                # Try to find JSON in response
                try:
                    brace_count = 0
                    start_pos = -1
                    for i, char in enumerate(response):
                        if char == '{':
                            if brace_count == 0:
                                start_pos = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_pos != -1:
                                json_str = response[start_pos:i + 1]
                                try:
                                    qa_pair = json.loads(json_str)
                                    if isinstance(qa_pair, dict) and "question" in qa_pair:
                                        qa_pairs.append(qa_pair)
                                except json.JSONDecodeError:
                                    pass
                                start_pos = -1
                except Exception:
                    pass
        return qa_pairs

    def process_text(self, text: str, source: str = "user_input"):
        """Process a single text to generate multi-hop QA pairs"""
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return {'qa_pairs': [], 'metadata': {'source': source, 'complexity': 0}}

        info_pairs = self._extract_info_pairs(processed_text)
        if info_pairs:
            qa_pairs = self._generate_qa_pairs(info_pairs)
        else:
            qa_pairs = []

        return {
            'qa_pairs': qa_pairs,
            'metadata': {
                'source': source,
                'complexity': len(qa_pairs),
            },
        }

    def process_batch(self, texts, sources=None):
        """Process multiple texts in batch"""
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        examples = []
        for text, source in tqdm(zip(texts, sources), total=len(texts), desc="Processing texts"):
            examples.append(self.process_text(text, source))
        return examples

    def forward_batch_input(
            self,
            data: List[dict],
            input_key: str = 'chunk_path',
            output_key: str = 'enhanced_chunk_path',
    ) -> List[dict]:
        """
        Generate multi-hop QA pairs from chunk files.

        Args:
            data: List of dict
            input_key: Key for input chunk file paths
            output_key: Key for output enhanced chunk file paths

        Returns:
            List of dict with enhanced chunk paths added
        """
        assert isinstance(data, list), "Input data must be a list of dict"

        chunk_paths = [item.get(input_key, "") for item in data]

        for chunk_path in chunk_paths:
            if chunk_path:
                texts = []
                if str(chunk_path).endswith(".json"):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    texts = [item.get("cleaned_chunk", "") for item in file_data]
                elif str(chunk_path).endswith(".jsonl"):
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        file_data = [json.loads(line) for line in f]
                    texts = [item.get("cleaned_chunk", "") for item in file_data]
                else:
                    LOG.warning(f"Unsupported file format: {chunk_path}")
                    continue

                # Generate QA pairs
                qa_pairs_batch = self.process_batch(texts)

                # Write back to original data
                for item, qa_result in zip(file_data, qa_pairs_batch):
                    item["qa_pairs"] = qa_result

                # Write back to file
                with open(chunk_path, "w", encoding="utf-8") as f:
                    if str(chunk_path).endswith(".json"):
                        json.dump(file_data, f, ensure_ascii=False, indent=4)
                    else:
                        for item in file_data:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                LOG.info(f"Generated multi-hop QA for {chunk_path}")

        results = []
        for item, chunk_path in zip(data, chunk_paths):
            new_item = item.copy()
            new_item[output_key] = chunk_path
            results.append(new_item)

        LOG.info("Multi-hop QA generation completed!")
        return results

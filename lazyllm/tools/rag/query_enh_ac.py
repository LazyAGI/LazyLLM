import json
from typing import List, Literal, Optional, Union

import lazyllm
from lazyllm.thirdparty import ahocorasick
from lazyllm import LOG, JsonFormatter, ChatPrompter
from lazyllm.module import LLMBase, TrainableModule

PromptLang = Literal['zh', 'en']

# ── LLM prompts (Chinese default, English optional) ───────────────────────────
_LLM_SYSTEM_ZH = '''你是一个自然语言处理专家，擅长判断词语边界与语义完整性。

## 任务
给定一段查询文本和若干候选匹配词，判断每个候选词在文本中是否以**完整、独立的词语**形式出现。

## 判断规则
1. 若候选词在文本中是一个语义完整的独立词语（不依附于更长的词），返回 true
2. 若候选词仅是文本中某个更长词语的组成部分（子串），返回 false

## 示例
- 文本"民法典"中的"民法" → false（"民法"是"民法典"的一部分）
- 文本"什么是民法？"中的"民法" → true（"民法"是完整独立的词语）
- 文本"加入丙醇酸"中的"丙醇" → false（"丙醇"是"丙醇酸"的一部分）
- 文本"加入丙醇和水"中的"丙醇" → true（"丙醇"是完整独立的词语）

## 输出格式
严格返回JSON数组，元素个数与候选词数量一致，每个元素为 true 或 false。
不要输出任何其他内容。
示例：[true, false, true]
'''

_LLM_USER_ZH = '查询文本："{query}"\n\n候选匹配：\n{candidates_text}'

_LLM_SYSTEM_EN = '''You are an NLP expert skilled at judging word boundaries and semantic completeness.

## Task
Given a query string and several candidate substring matches,
decide whether each candidate appears as a **complete, standalone word** in the text.

## Rules
1. If the candidate is a semantically complete word on its own (not part of a longer word), return true.
2. If the candidate is only a substring inside a longer word, return false.

## Examples
- In "民法典", candidate "民法" → false ("民法" is part of "民法典").
- In "什么是民法？", candidate "民法" → true ("民法" is a standalone word).
- In "加入丙醇酸", candidate "丙醇" → false ("丙醇" is part of "丙醇酸").
- In "加入丙醇和水", candidate "丙醇" → true ("丙醇" is a standalone word).

## Output
Return **only** a JSON array with one boolean per candidate, same order as listed.
Example: [true, false, true]
'''

_LLM_USER_EN = 'Query text: "{query}"\n\nCandidates:\n{candidates_text}'


def _default_llm_prompt(lang: PromptLang) -> ChatPrompter:
    if lang == 'en':
        return ChatPrompter({'system': _LLM_SYSTEM_EN, 'user': _LLM_USER_EN})
    return ChatPrompter({'system': _LLM_SYSTEM_ZH, 'user': _LLM_USER_ZH})


class _LLMFilter:

    _default_inference_kwargs: dict = {}

    def __init__(
        self,
        model: LLMBase,
        prompt=None,
        max_retries: int = 3,
        prompt_lang: PromptLang = 'zh',
    ):
        lang: PromptLang = prompt_lang if prompt_lang in ('zh', 'en') else 'zh'
        self._prompt_lang = lang
        prompt = prompt if prompt is not None else _default_llm_prompt(lang)
        self.model = model.share().prompt(prompt).formatter(JsonFormatter())
        self._max_retries = max_retries

    def _preprocess(self, query: str, matches: list) -> dict:
        if self._prompt_lang == 'en':
            lines = [
                f'{i}. matched_word="{m["word"]}", '
                f'context="...{query[max(0, m["start"] - 5):m["end"] + 6]}..."'
                for i, m in enumerate(matches)
            ]
        else:
            lines = [
                f'{i}. 匹配词="{m["word"]}", '
                f'上下文="...{query[max(0, m["start"] - 5):m["end"] + 6]}..."'
                for i, m in enumerate(matches)
            ]
        candidates_text = '\n'.join(lines)
        return {'query': query, 'candidates_text': candidates_text}

    def __call__(self, query: str, matches: list) -> list:
        if not matches:
            return []
        prepared = self._preprocess(query, matches)
        for attempt in range(self._max_retries):
            try:
                res = self.model(prepared, **self._default_inference_kwargs)
                if isinstance(res, list) and len(res) == len(matches):
                    return [m for m, keep in zip(matches, res) if keep]
                LOG.warning(
                    f'_LLMFilter invalid output, attempt {attempt + 1}/{self._max_retries}: {res}'
                )
            except Exception as e:
                LOG.warning(
                    f'_LLMFilter inference failed, attempt {attempt + 1}/{self._max_retries}: {e}'
                )
        LOG.warning(
            '_LLMFilter gave up after retries; skipping enhancement (original query unchanged).'
        )
        return []


def _is_bert_deploy(module: TrainableModule) -> bool:
    return module._deploy_type is lazyllm.deploy.BertDeploy


class _BERTFilter:

    def __init__(
        self,
        model: TrainableModule,
        threshold: float = 0.5,
        max_retries: int = 3,
    ):
        self.model = model.share()
        self.threshold = float(threshold)
        self._max_retries = max_retries

    @staticmethod
    def _context(query: str, match: dict) -> str:
        s, e = match['start'], match['end']
        return f'...{query[max(0, s - 5):e + 6]}...'

    @staticmethod
    def _prob_label1(obj: dict) -> float:
        probs = obj.get('probs') or []
        if len(probs) >= 2:
            return float(probs[1])
        if len(probs) == 1:
            return float(probs[0])
        raise ValueError(f'BERT response missing probs: {obj!r}')

    @classmethod
    def _parse_output(cls, raw: Union[str, dict]) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        raise TypeError(f'Unexpected BERT output type: {type(raw)}')

    def __call__(self, query: str, matches: list) -> list:
        if not matches:
            return matches

        kept: list = []
        for m in matches:
            word, ctx = m['word'], self._context(query, m)
            placed = False
            for attempt in range(self._max_retries):
                try:
                    raw = self.model(word, text_b=ctx)
                    obj = self._parse_output(raw)
                    if self._prob_label1(obj) >= self.threshold:
                        kept.append(m)
                    placed = True
                    break
                except Exception as e:
                    LOG.warning(
                        f'_BERTFilter inference failed, attempt {attempt + 1}/{self._max_retries}: {e}'
                    )
            if not placed:
                LOG.warning(
                    f'_BERTFilter gave up on match after retries: {m.get("word")!r}; dropping that match only.'
                )

        return kept


class QueryEnhACProcessor:

    def __init__(
        self,
        data_source=None,
        discriminator=None,
        cluster_key: str = 'cluster_id',
        word_key: str = 'word',
        max_retries: int = 3,
        prompt_lang: PromptLang = 'zh',
    ):
        if prompt_lang not in ('zh', 'en'):
            raise ValueError(f'prompt_lang must be "zh" or "en", got {prompt_lang!r}')
        self.cluster_key = cluster_key
        self.word_key = word_key
        self._max_retries = max_retries
        self._prompt_lang: PromptLang = prompt_lang

        self._boundary_filter: Optional[Union[_LLMFilter, _BERTFilter]] = None
        self._build_filter(discriminator)

        self.vocab_data = []
        self.word_to_cluster = {}
        self.cluster_to_words = {}
        self.automaton = None

        self._data_source = data_source if data_source is not None else []
        self._rebuild_automaton()

    def _build_filter(self, discriminator):
        if discriminator is None:
            self._boundary_filter = None
            return
        if isinstance(discriminator, TrainableModule) and _is_bert_deploy(discriminator):
            self._boundary_filter = _BERTFilter(
                model=discriminator,
                max_retries=self._max_retries,
            )
            return
        if isinstance(discriminator, LLMBase):
            self._boundary_filter = _LLMFilter(
                model=discriminator,
                max_retries=self._max_retries,
                prompt_lang=self._prompt_lang,
            )
            return
        raise TypeError(
            f'Unsupported discriminator type {type(discriminator)}. '
            'Use OnlineChatModule, TrainableModule (LLM or lazyllm.deploy.BertDeploy), or None.'
        )

    def _rebuild_automaton(self):
        if callable(self._data_source):
            self.vocab_data = self._data_source()
        else:
            self.vocab_data = list(self._data_source)

        self.word_to_cluster = {}
        self.cluster_to_words = {}

        for item in self.vocab_data:
            cluster_id = item.get(self.cluster_key)
            word = item.get(self.word_key)
            if cluster_id is None or word is None:
                continue
            self.word_to_cluster[word] = cluster_id
            if cluster_id not in self.cluster_to_words:
                self.cluster_to_words[cluster_id] = []
            if word not in self.cluster_to_words[cluster_id]:
                self.cluster_to_words[cluster_id].append(word)

        if self.word_to_cluster:
            automaton = ahocorasick.Automaton()
            for word, cluster_id in self.word_to_cluster.items():
                automaton.add_word(str(word), (cluster_id, str(word)))
            automaton.make_automaton()
            self.automaton = automaton
        else:
            self.automaton = None

        LOG.info(f'AC automaton built, vocabulary size: {len(self.word_to_cluster)}')

    def update_data_source(self, data_source):
        self._data_source = data_source
        self._rebuild_automaton()

    def update_discriminator(self, discriminator):
        self._build_filter(discriminator)

    def _get_matches(self, query: str) -> list:
        if not self.automaton or not query:
            return []

        raw_matches = []
        for end_idx, (cluster_id, matched_word) in self.automaton.iter(str(query)):
            start_idx = end_idx - len(matched_word) + 1
            raw_matches.append({
                'word': matched_word,
                'cluster_id': cluster_id,
                'start': start_idx,
                'end': end_idx,
            })

        if not raw_matches:
            return []

        if self._boundary_filter is None:
            LOG.warning(
                'QueryEnhACProcessor: discriminator is None but AC automaton had matches; '
                'skipping enhancement (original query unchanged).'
            )
            return []

        raw_matches.sort(key=lambda x: (x['start'], -len(x['word'])))
        result, last_end = [], -1
        for m in raw_matches:
            if m['start'] <= last_end:
                continue
            result.append(m)
            last_end = m['end']
        return self._boundary_filter(query, result)

    def _enhance_single(self, query: str) -> str:
        matches = self._get_matches(query)
        if not matches:
            return query

        enhanced_parts = []
        last_pos = 0
        seen_clusters: set = set()

        for match in matches:
            start_idx = match['start']
            end_idx = match['end']

            if start_idx > last_pos:
                enhanced_parts.append(query[last_pos:start_idx])

            cluster_id = match['cluster_id']
            if cluster_id in seen_clusters:
                replacement = match['word']
            else:
                seen_clusters.add(cluster_id)
                cluster_words = self.cluster_to_words.get(cluster_id, [])
                other_words = [w for w in cluster_words if w != match['word']]
                replacement = (
                    f'{match["word"]}（{", ".join(other_words)}）'
                    if other_words else match['word']
                )

            enhanced_parts.append(replacement)
            last_pos = end_idx + 1

        if last_pos < len(query):
            enhanced_parts.append(query[last_pos:])

        return ''.join(enhanced_parts)

    def get_matches(self, query: str) -> List[dict]:
        out: List[dict] = []
        for m in self._get_matches(query):
            cid = m['cluster_id']
            out.append({
                self.word_key: m['word'],
                self.cluster_key: cid,
                'cluster_words': self.cluster_to_words.get(cid, []),
            })
        return out

    def __call__(self, queries: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(queries, str):
            return self._enhance_single(queries)
        return [self._enhance_single(q) for q in queries]

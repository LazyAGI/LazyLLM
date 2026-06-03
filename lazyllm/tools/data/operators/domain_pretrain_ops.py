import re
import math
import unicodedata
from collections import Counter
from typing import List, Optional, Dict

from ..base_data import data_register
from lazyllm import LOG
from lazyllm.thirdparty import jieba

from lazyllm.common.registry import LazyLLMRegisterMetaClass

if 'data' in LazyLLMRegisterMetaClass.all_clses and 'domain_pretrain' in LazyLLMRegisterMetaClass.all_clses['data']:
    domain_pretrain = LazyLLMRegisterMetaClass.all_clses['data']['domain_pretrain'].base
else:
    domain_pretrain = data_register.new_group('domain_pretrain')


class PretrainFieldNormalizer(domain_pretrain):
    _DEFAULT_TEXT_FIELDS = (
        'content', 'text', 'article', 'document', 'body', 'passage',
        'abstract', 'description', 'summary', 'note', 'report',
        'sentence', 'paragraph', 'raw_text', 'full_text',
        'clinical_note', 'diagnosis', 'medical_text',
        'output', 'answer', 'response', 'message', 'query', 'value',
    )

    def __init__(
        self,
        content_key: str = 'content',
        field_mapping: Optional[Dict[str, str]] = None,
        concat_fields: Optional[List[str]] = None,
        concat_separator: str = '\n\n',
        fallback_fields: Optional[List[str]] = None,
        drop_original: bool = False,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.content_key = content_key
        self.field_mapping = field_mapping or {}
        self.concat_fields = concat_fields
        self.concat_separator = concat_separator
        self.fallback_fields = fallback_fields or list(self._DEFAULT_TEXT_FIELDS)
        self.drop_original = drop_original

    def _apply_field_mapping(self, data: dict) -> dict:
        for src, dst in self.field_mapping.items():
            if src in data and src != dst:
                data[dst] = data.pop(src) if self.drop_original else data[src]
        return data

    def _try_concat(self, data: dict) -> bool:
        if not self.concat_fields:
            return False
        parts = []
        for key in self.concat_fields:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if not parts:
            return False
        data[self.content_key] = self.concat_separator.join(parts)
        if self.drop_original:
            for key in self.concat_fields:
                if key != self.content_key and key in data:
                    data.pop(key)
        return True

    def _try_fallback(self, data: dict) -> bool:
        for key in self.fallback_fields:
            if key == self.content_key:
                continue
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                data[self.content_key] = val.strip()
                if self.drop_original and key in data:
                    data.pop(key)
                return True
        return False

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)

        data = self._apply_field_mapping(data)

        if self.content_key in data and isinstance(data[self.content_key], str) and data[self.content_key].strip():
            return data

        if self._try_concat(data):
            return data

        if self._try_fallback(data):
            return data

        LOG.debug(f'PretrainFieldNormalizer: no text field found in keys={list(data.keys())}')
        return None


class DomainKeywordFilter(domain_pretrain):
    def __init__(
        self,
        input_key: str = 'content',
        keywords: Optional[List[str]] = None,
        mode: str = 'any',
        min_keyword_hits: int = 1,
        min_keyword_density: float = 0.001,
        case_sensitive: bool = False,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.keywords = keywords or []
        self.mode = mode
        self.min_keyword_hits = min_keyword_hits
        self.min_keyword_density = min_keyword_density
        self.case_sensitive = case_sensitive
        self._automaton = None

    def _build_automaton(self):
        if self._automaton is not None:
            return
        try:
            from lazyllm.thirdparty import ahocorasick
            self._automaton = ahocorasick.Automaton()
            for idx, kw in enumerate(self.keywords):
                key = kw if self.case_sensitive else kw.lower()
                self._automaton.add_word(key, (idx, kw))
            self._automaton.make_automaton()
        except Exception:
            LOG.warning('ahocorasick not available, falling back to regex matching')
            self._automaton = None

    def _match_with_automaton(self, text: str):
        hits = Counter()
        search_text = text if self.case_sensitive else text.lower()
        for _, (_, kw) in self._automaton.iter(search_text):
            hits[kw] += 1
        return hits

    def _match_with_regex(self, text: str):
        hits = Counter()
        search_text = text if self.case_sensitive else text.lower()
        for kw in self.keywords:
            key = kw if self.case_sensitive else kw.lower()
            count = search_text.count(key)
            if count > 0:
                hits[kw] = count
        return hits

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if not isinstance(text, str) or not text.strip():
            return None
        if not self.keywords:
            return data

        self._build_automaton()
        hits = self._match_with_automaton(text) if self._automaton else self._match_with_regex(text)

        total_hits = sum(hits.values())

        if self.mode == 'density':
            char_count = max(len(text), 1)
            density = total_hits / char_count
            if density < self.min_keyword_density:
                return None
        else:
            if total_hits < self.min_keyword_hits:
                return None

        data['_keyword_hits'] = total_hits
        data['_keyword_types'] = len(hits)
        return data


class DomainRelevanceScorer(domain_pretrain):
    def __init__(
        self,
        input_key: str = 'content',
        keywords: Optional[List[str]] = None,
        keyword_weights: Optional[Dict[str, float]] = None,
        min_score: float = 0.1,
        language: str = 'zh',
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.keywords = keywords or []
        self.keyword_weights = keyword_weights or {}
        self.min_score = min_score
        self.language = language.lower()

    def _tokenize(self, text: str) -> List[str]:
        if self.language in ('zh', 'cn', 'chinese'):
            return list(jieba.cut(text))
        return text.lower().split()

    def _compute_score(self, text: str) -> float:
        if not self.keywords:
            return 1.0
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0

        total_tokens = len(tokens)
        token_counter = Counter(tokens)
        score = 0.0
        for kw in self.keywords:
            tf = token_counter.get(kw, 0) / total_tokens
            weight = self.keyword_weights.get(kw, 1.0)
            idf = math.log(1 + 1.0 / max(weight, 0.01))
            score += tf * idf * weight

        return score

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if not isinstance(text, str) or not text.strip():
            return None

        score = self._compute_score(text)
        if score < self.min_score:
            return None

        data['_domain_relevance_score'] = round(score, 6)
        return data


class NGramRepetitionFilter(domain_pretrain):
    def __init__(
        self,
        input_key: str = 'content',
        n: int = 10,
        max_repetition_ratio: float = 0.3,
        language: str = 'zh',
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.n = n
        self.max_repetition_ratio = max_repetition_ratio
        self.language = language.lower()

    def _tokenize(self, text: str) -> List[str]:
        if self.language in ('zh', 'cn', 'chinese'):
            return list(jieba.cut(text))
        return text.lower().split()

    def _compute_repetition_ratio(self, tokens: List[str]) -> float:
        if len(tokens) < self.n:
            return 0.0

        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        ngram_counts = Counter(ngrams)
        repeated = sum(count for count in ngram_counts.values() if count > 1)
        return repeated / len(ngrams)

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if not isinstance(text, str) or not text.strip():
            return None

        tokens = self._tokenize(text)
        ratio = self._compute_repetition_ratio(tokens)

        if ratio > self.max_repetition_ratio:
            LOG.debug(f'Filtered by n-gram repetition: ratio={ratio:.3f} > {self.max_repetition_ratio}')
            return None

        data['_ngram_repetition_ratio'] = round(ratio, 4)
        return data


_PHONE_PATTERNS = [
    r'(?<!\d)1[3-9]\d{9}(?!\d)',
    r'(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)',
    r'\+86[-\s]?1[3-9]\d{9}',
    r'\+\d{1,3}[-\s]?\(?\d{1,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}',
]
_EMAIL_PATTERN = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
_IP_PATTERN = r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)'
_BANK_CARD_PATTERN = r'(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?:\d{0,3})?(?!\d)'

_COMPILED_PHONE = [re.compile(p) for p in _PHONE_PATTERNS]
_COMPILED_EMAIL = re.compile(_EMAIL_PATTERN)
_COMPILED_IP = re.compile(_IP_PATTERN)
_COMPILED_BANK_CARD = re.compile(_BANK_CARD_PATTERN)


@data_register('data.domain_pretrain', rewrite_func='forward', _concurrency_mode='process')
def sensitive_info_cleaner(
    data: dict,
    input_key: str = 'content',
    remove_phone: bool = True,
    remove_email: bool = True,
    remove_ip: bool = True,
    remove_bank_card: bool = True,
    replacement: str = '[REDACTED]',
):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if not isinstance(text, str):
        return data

    if remove_phone:
        for pattern in _COMPILED_PHONE:
            text = pattern.sub(replacement, text)
    if remove_email:
        text = _COMPILED_EMAIL.sub(replacement, text)
    if remove_ip:
        text = _COMPILED_IP.sub(replacement, text)
    if remove_bank_card:
        text = _COMPILED_BANK_CARD.sub(replacement, text)

    data[input_key] = text
    return data


@data_register('data.domain_pretrain', rewrite_func='forward', _concurrency_mode='process')
def text_normalizer(
    data: dict,
    input_key: str = 'content',
    unicode_form: str = 'NFKC',
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    strip: bool = True,
):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if not isinstance(text, str):
        return data

    if fix_encoding:
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)

    if normalize_whitespace:
        text = re.sub(r'[\t\r]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

    if strip:
        text = text.strip()

    data[input_key] = text
    return data


class DocumentLanguageFilter(domain_pretrain):
    _CJK_RANGES = (
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0xF900, 0xFAFF),
    )

    def __init__(
        self,
        input_key: str = 'content',
        target_language: str = 'zh',
        min_target_ratio: float = 0.3,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.target_language = target_language.lower()
        self.min_target_ratio = min_target_ratio

    def _cjk_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        cjk_count = sum(
            1 for ch in text
            if any(lo <= ord(ch) <= hi for lo, hi in self._CJK_RANGES)
        )
        alpha_count = sum(1 for ch in text if ch.isalpha())
        return cjk_count / max(alpha_count, 1)

    def _ascii_alpha_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        ascii_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        alpha_count = sum(1 for ch in text if ch.isalpha())
        return ascii_count / max(alpha_count, 1)

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if not isinstance(text, str) or not text.strip():
            return None

        if self.target_language in ('zh', 'cn', 'chinese'):
            ratio = self._cjk_ratio(text)
        elif self.target_language in ('en', 'english'):
            ratio = self._ascii_alpha_ratio(text)
        else:
            return data

        if ratio < self.min_target_ratio:
            return None

        data['_language_ratio'] = round(ratio, 4)
        return data

__all__ = [
    'PretrainFieldNormalizer',
    'DomainKeywordFilter',
    'DomainRelevanceScorer',
    'NGramRepetitionFilter',
    'sensitive_info_cleaner',
    'text_normalizer',
    'DocumentLanguageFilter',
]

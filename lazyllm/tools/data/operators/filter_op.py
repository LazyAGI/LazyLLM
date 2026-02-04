import os
import re
from ..base_data import DataOperatorRegistry
from lazyllm import LOG, config

import fasttext
from huggingface_hub import hf_hub_download
from datasketch import MinHash, MinHashLSH
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
import jieba


@DataOperatorRegistry.register
class LanguageFilter:
    COMMON_LANGUAGES = {
        'zho_Hans', 'zho_Hant', 'eng_Latn', 'spa_Latn', 'fra_Latn',
        'deu_Latn', 'jpn', 'kor', 'rus_Cyrl', 'ara', 'por_Latn',
        'ita_Latn', 'nld_Latn', 'pol_Latn', 'tur_Latn', 'vie',
        'tha', 'hin', 'ind_Latn', 'msa_Latn'
    }

    def __init__(self, input_key='content', target_language='zh', threshold=0.6, model_path=None, model_cache_dir=None):
        self.input_key = input_key
        if isinstance(target_language, str):
            self.allowed_languages = {target_language}
        else:
            self.allowed_languages = set(target_language)
        self.threshold = threshold
        self.model_path = model_path
        self.model_cache_dir = model_cache_dir or config['model_cache_dir']
        self._validate_languages()
        self.model = self._load_model()

    def _validate_languages(self):
        invalid_langs = self.allowed_languages - self.COMMON_LANGUAGES
        if invalid_langs:
            LOG.warning(
                f'LanguageFilter: Invalid language codes: {invalid_langs}\n'
                f'Common language codes:\n'
                f'  - zho_Hans (Simplified Chinese), zho_Hant (Traditional Chinese)\n'
                f'  - eng_Latn (English)\n'
                f'  - spa_Latn (Spanish), fra_Latn (French), deu_Latn (German)\n'
                f'  - jpn (Japanese), kor (Korean)\n'
                f'  - rus_Cyrl (Russian), ara (Arabic)\n'
                f'  - por_Latn (Portuguese), ita_Latn (Italian)\n'
                f'Full list: {sorted(self.COMMON_LANGUAGES)}'
            )

    def _load_model(self):
        try:
            if self.model_path:
                # Use provided model path directly if available
                LOG.info(f'Loading FastText language model from {self.model_path}...')
                model = fasttext.load_model(self.model_path)
            else:
                # Download model to cache directory
                LOG.info('Downloading FastText language identification model from Hugging Face Hub...')
                model_path = hf_hub_download(
                    repo_id='facebook/fasttext-language-identification',
                    filename='model.bin',
                    cache_dir=self.model_cache_dir
                )
                model = fasttext.load_model(model_path)
            LOG.info('FastText language model loaded successfully.')
            return model
        except Exception as e:
            LOG.error(f'Error loading FastText model: {e}')
            raise

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        k = max(5, len(self.allowed_languages))
        labels, scores = self.model.predict(text.replace('\n', ' ').strip(), k=k)
        if len(labels) > 0 and len(scores) > 0:
            for label, score in zip(labels, scores):
                pred_label = label.replace('__label__', '')
                if pred_label in self.allowed_languages and score >= self.threshold:
                    return data

        return None


@DataOperatorRegistry.register(one_item=False)
class MinHashDeduplicateFilter:
    def __init__(self, input_key='content', threshold=0.85, num_perm=128, use_n_gram=True, ngram=5):
        self.input_key = input_key
        self.threshold = threshold
        self.num_perm = num_perm
        self.use_n_gram = use_n_gram
        self.ngram = ngram
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._item_counter = 0
        self._minhash_map = {}

    def _create_minhash(self, text):
        minhash = MinHash(num_perm=self.num_perm)
        if self.use_n_gram:
            for i in range(len(text) - self.ngram + 1):
                minhash.update(text[i:i + self.ngram].encode('utf8'))
        else:
            for char in text:
                minhash.update(char.encode('utf8'))
        return minhash

    def __call__(self, data):
        assert isinstance(data, list)

        kept_items = []
        for item in data:
            if not isinstance(item, dict) or self.input_key not in item:
                continue

            text = item[self.input_key]
            if not isinstance(text, str) or not text.strip():
                continue

            minhash = self._create_minhash(text)
            result = self.lsh.query(minhash)

            if len(result) == 0:
                self.lsh.insert(self._item_counter, minhash)
                self._minhash_map[self._item_counter] = minhash
                self._item_counter += 1
                kept_items.append(item)

        return kept_items


@DataOperatorRegistry.register
class BlocklistFilter:
    def __init__(self, input_key='content', blocklist=None, blocklist_path=None,
                 language='en', threshold=1, use_tokenizer=False):
        self.input_key = input_key
        self.threshold = threshold
        self.use_tokenizer = use_tokenizer
        self.language = language.lower()

        if self.use_tokenizer and self.language in ['en', 'english']:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                LOG.info('Downloading NLTK punkt_tab tokenizer...')
                nltk.download('punkt_tab')

        if blocklist is not None:
            self.blocklist = set(word.strip().lower() for word in blocklist)
        elif blocklist_path is not None:
            self.blocklist = self._load_blocklist_from_file(blocklist_path)
        else:
            raise ValueError('Either blocklist or blocklist_path must be provided')

        LOG.info(f'BlocklistFilter initialized with {len(self.blocklist)} blocked words, '
                 f'language={self.language}, use_tokenizer={self.use_tokenizer}')

    def _load_blocklist_from_file(self, file_path):
        LOG.info(f'Loading blocklist from {file_path}...')
        with open(file_path, 'r', encoding='utf-8') as f:
            blocklist = set(line.strip().lower() for line in f if line.strip())
        LOG.info(f'Loaded {len(blocklist)} words from blocklist')
        return blocklist

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return data

        if self.use_tokenizer:
            if self.language in ['zh', 'cn', 'chinese']:
                words = list(jieba.cut(text.lower()))
            elif self.language in ['en', 'english']:
                words = word_tokenize(text.lower())
            else:
                LOG.warning(f'Unsupported language: {self.language}, using simple split')
                words = text.lower().split()
        else:
            words = text.lower().split()

        blocklist_count = sum(1 for word in words if word in self.blocklist)

        if blocklist_count <= self.threshold:
            return data
        else:
            return None


@DataOperatorRegistry.register
class WordCountFilter:
    def __init__(self, input_key='content', min_words=10, max_words=10000, language='zh'):
        self.input_key = input_key
        self.min_words = min_words
        self.max_words = max_words
        self.language = language.lower()

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        if self.language in ['zh', 'cn', 'chinese']:
            count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        elif self.language in ['en', 'english']:
            count = len(text.split())
        else:
            LOG.warning(f'Unsupported language: {self.language}, using character count')
            count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))

        if self.min_words <= count < self.max_words:
            return data
        else:
            return None


@DataOperatorRegistry.register
class ColonEndFilter:
    def __init__(self, input_key='content'):
        self.input_key = input_key

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return data

        if text.rstrip().endswith(':') or text.rstrip().endswith('：'):
            return None
        else:
            return data


@DataOperatorRegistry.register
class SentenceCountFilter:
    def __init__(self, input_key='content', min_sentences=3, max_sentences=1000, language='zh'):
        self.input_key = input_key
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.language = language.lower()

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        if self.language in ['zh', 'cn', 'chinese']:
            sentences = re.split(r'[。！？]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            num_sentences = len(sentences)
        elif self.language in ['en', 'english']:
            sentences = sent_tokenize(text)
            num_sentences = len(sentences)
        else:
            LOG.warning(f'Unsupported language: {self.language}, using Chinese punctuation')
            sentences = re.split(r'[。！？]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            num_sentences = len(sentences)

        if self.min_sentences <= num_sentences <= self.max_sentences:
            return data
        else:
            return None


@DataOperatorRegistry.register
class EllipsisEndFilter:
    def __init__(self, input_key='content', max_ratio=0.3):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.ellipsis = ['...', '…', '……']

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return data

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        num_lines = len(lines)

        if num_lines == 0:
            return data

        num_occurrences = sum(
            1 for line in lines
            if any(line.endswith(ellipsis) for ellipsis in self.ellipsis)
        )
        ratio = num_occurrences / num_lines

        if ratio < self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class NullContentFilter:
    def __init__(self, input_key='content'):
        self.input_key = input_key

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        
        if text is not None and isinstance(text, str) and text.strip() != '':
            return data
        else:
            return None


@DataOperatorRegistry.register
class WordLengthFilter:
    def __init__(self, input_key='content', min_length=3, max_length=20):
        self.input_key = input_key
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return None

        num_chars = sum(len(word) for word in words)
        mean_length = num_chars / num_words
        if self.min_length <= mean_length < self.max_length:
            return data
        else:
            return None


@DataOperatorRegistry.register
class SymbolRatioFilter:
    def __init__(self, input_key='content', max_ratio=0.3, symbols=None):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.symbols = symbols or ['#', '...', '…']
        self.tokenizer = WordPunctTokenizer()

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        tokens = self.tokenizer.tokenize(text)
        word_tokens = [t for t in tokens if t not in self.symbols]
        num_words = len(word_tokens)
        if num_words == 0:
            return None

        num_symbols = sum(text.count(symbol) for symbol in self.symbols)
        ratio = num_symbols / num_words
        if ratio < self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class IDCardFilter:
    def __init__(self, input_key='content', threshold=3):
        self.input_key = input_key
        self.threshold = threshold
        self.chinese_terms = [
            r'身\s{0,5}份\s{0,5}证(?:\s{0,5}号(?:\s{0,5}码)?)?',
            r'证\s{0,5}件(?:\s{0,5}号(?:\s{0,5}码)?)?',
            r'居\s{0,5}民\s{0,5}身\s{0,5}份\s{0,5}证',
            r'个\s{0,5}人\s{0,5}身\s{0,5}份',
        ]

        self.english_terms = [
            r'id\s{0,10}number',
            r'id\s{0,10}card',
            r'id\s{0,10}no\.?',
            r'i\.?\s{0,5}d\.?\s{0,10}number',
            r'identification(?:\s{0,10}number)?(?:\s{0,10}card)?',
            r'identity(?:\s{0,10}card)?(?:\s{0,10}number)?',
            r'national\s{0,10}id(?:\s{0,10}card)?',
            r'government\s{0,10}id(?:\s{0,10}card)?',
            r'nric(?:\s{0,10}number)?',
            r'ic\s{0,10}number',
            r'resident\s{0,10}registration(?:\s{0,10}number)?',
            r'personal\s{0,10}id(?:\s{0,10}number)?',
            r'passport\s{0,10}number',
            r'social\s{0,10}security\s{0,10}number',
            r'ssn',
        ]

        all_patterns = self.chinese_terms + self.english_terms
        self.pattern = re.compile('|'.join(f'({p})' for p in all_patterns), re.I)

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        matches = self.pattern.findall(text)
        has_too_many_id_terms = len(matches) >= self.threshold
        if not has_too_many_id_terms:
            return data
        else:
            return None


@DataOperatorRegistry.register
class NoPunctuationFilter:
    def __init__(self, input_key='content', max_length_between_punct=112, language='zh'):
        self.input_key = input_key
        self.max_length_between_punct = max_length_between_punct
        self.language = language.lower()

        if self.language in ['zh', 'cn', 'chinese']:
            self.punct_pattern = r'[。！？；，、：""''（）《》【】…—\.\!\?,;:]'
        elif self.language in ['en', 'english']:
            self.punct_pattern = r'[–.!?,;•/|…:;\'\"]'
        else:
            LOG.warning(f'Unsupported language: {self.language}, using Chinese punctuation')
            self.punct_pattern = r'[。！？；，、：""''（）《》【】…—\.\!\?,;:]'

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        paragraphs = text.split('\n')
        max_length = 0

        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0:
                continue

            segments = re.split(self.punct_pattern, paragraph)
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                if self.language in ['en', 'english']:
                    length = len(segment.split())
                else:
                    length = len(segment)

                if length > max_length:
                    max_length = length

        if max_length <= self.max_length_between_punct:
            return data
        else:
            return None


@DataOperatorRegistry.register
class SpecialCharFilter:
    def __init__(self, input_key='content'):
        self.input_key = input_key
        self.special_patterns = [
            r'\u200b|\u200c|\u200d|\u200e|\u200f',
            r'\ufeff',
            r'\u2060|\u2061|\u2062|\u2063',
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',
            r'[\x80-\x9f]',
            r'[�□�]',
            r'\ufffd',
            r'&#\d{1,5};',
            r'&[a-zA-Z]{2,10};',
            r'\? :',
            r'\{\/U\}',
            r'U\+[0-9A-Fa-f]{4,6}',
            r'\\u[0-9A-Fa-f]{4}',
            r'\\x[0-9A-Fa-f]{2}',
            r'U\+1F[0-9A-Fa-f]{3}',
            r'U\+26[0-9A-Fa-f]{2}',
            r'U\+27[0-9A-Fa-f]{2}',
            r'\u202a|\u202b|\u202c|\u202d|\u202e',
            r'[\ue000-\uf8ff]',
            r'\ud800-\udfff',
        ]

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        has_special_char = any(re.search(pattern, text) for pattern in self.special_patterns)
        if not has_special_char:
            return data
        else:
            return None


@DataOperatorRegistry.register
class WatermarkFilter:
    def __init__(self, input_key='content', watermarks=None):
        self.input_key = input_key
        self.watermarks = watermarks or [
            'Copyright', 'Watermark', 'Confidential', 'All Rights Reserved',
            'Proprietary', 'Trade Secret', 'Internal Use Only', 'Do Not Distribute',
            'Private and Confidential', 'Restricted', 'Classified', 'Top Secret',
            '版权所有', '保留所有权利', '机密', '内部资料', '严禁转载', '禁止复制',
            '仅供内部使用', '未经授权', '商业机密', '翻版必究'
        ]

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        matches = re.search('|'.join(self.watermarks), text)
        if matches is None:
            return data
        else:
            return None


@DataOperatorRegistry.register
class StopWordFilter:
    def __init__(self, input_key='content', max_ratio=0.5, use_tokenizer=True, language='zh'):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.use_tokenizer = use_tokenizer
        self.language = language.lower()

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            LOG.info('Downloading NLTK stopwords...')
            nltk.download('stopwords')

        if self.language in ['en', 'english']:
            self.stopwords = set(stopwords.words('english'))
        elif self.language in ['zh', 'cn', 'chinese']:
            self.stopwords = set(stopwords.words('chinese'))
        else:
            LOG.warning(f'Unsupported language: {self.language}, using English stopwords')
            self.stopwords = set(stopwords.words('english'))

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        if self.language in ['zh', 'cn', 'chinese']:
            if self.use_tokenizer:
                words = list(jieba.cut(text.lower()))
            else:
                words = list(text)
        elif self.language in ['en', 'english']:
            if self.use_tokenizer:
                words = word_tokenize(text.lower())
            else:
                words = text.lower().split()
        else:
            words = text.lower().split()

        num_words = len(words)
        if num_words == 0:
            return None

        num_stop_words = sum(1 for w in words if w in self.stopwords)
        ratio = num_stop_words / num_words

        if ratio < self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class CurlyBracketFilter:
    def __init__(self, input_key='content', max_ratio=0.025):
        self.input_key = input_key
        self.max_ratio = max_ratio

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        num_brackets = text.count('{') + text.count('}')
        ratio = num_brackets / len(text) if len(text) > 0 else 0
        if ratio < self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class CapitalWordFilter:
    def __init__(self, input_key='content', max_ratio=0.2, use_tokenizer=False):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.use_tokenizer = use_tokenizer

        if self.use_tokenizer:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                LOG.info('Downloading NLTK punkt_tab tokenizer...')
                nltk.download('punkt_tab')

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        if self.use_tokenizer:
            words = word_tokenize(text)
        else:
            words = text.split()

        num_words = len(words)
        if num_words == 0:
            return None

        num_caps_words = sum(1 for word in words if word.isupper())
        ratio = num_caps_words / num_words

        if ratio <= self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class LoremIpsumFilter:
    def __init__(self, input_key='content', max_ratio=3e-8):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.pattern = re.compile(r'lorem ipsum', re.IGNORECASE)

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        num_occurrences = len(self.pattern.findall(text))
        ratio = num_occurrences / len(text) if len(text) > 0 else 0

        if ratio <= self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class UniqueWordFilter:
    def __init__(self, input_key='content', min_ratio=0.1):
        self.input_key = input_key
        self.min_ratio = min_ratio

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        words = text.lower().split()
        num_words = len(words)

        if num_words == 0:
            return None

        num_unique_words = len(set(words))
        ratio = num_unique_words / num_words

        if ratio > self.min_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class CharCountFilter:
    def __init__(self, input_key='content', min_chars=100, max_chars=100000):
        self.input_key = input_key
        self.min_chars = min_chars
        self.max_chars = max_chars

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        text_no_space = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
        num_chars = len(text_no_space)

        if self.min_chars <= num_chars <= self.max_chars:
            return data
        else:
            return None


@DataOperatorRegistry.register
class BulletPointFilter:
    def __init__(self, input_key='content', max_ratio=0.9):
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.bullet_chars = [
            '\u2022', '\u2023', '\u25B6', '\u25C0', '\u25E6',
            '\u25A0', '\u25A1', '\u25AA', '\u25AB', '\u2013'
        ]

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        num_lines = len(lines)

        if num_lines == 0:
            return None

        num_bullet_lines = sum(
            1 for line in lines
            if any(line.startswith(bullet) for bullet in self.bullet_chars)
        )
        ratio = num_bullet_lines / num_lines

        if ratio <= self.max_ratio:
            return data
        else:
            return None


@DataOperatorRegistry.register
class JavascriptFilter:
    def __init__(self, input_key='content', min_non_js_lines=3):
        self.input_key = input_key
        self.min_non_js_lines = min_non_js_lines

    def __call__(self, data):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return None

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        num_lines = len(lines)

        if num_lines == 0:
            return None

        num_js_lines = sum(1 for line in lines if 'javascript' in line.lower())
        num_non_js_lines = num_lines - num_js_lines

        if num_lines <= 3 or num_non_js_lines >= self.min_non_js_lines:
            return data
        else:
            return None

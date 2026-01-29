import os
from ..base_data import DataOperatorRegistry
from lazyllm import LOG, config

import fasttext
from huggingface_hub import hf_hub_download
from datasketch import MinHash, MinHashLSH
import nltk
from nltk.tokenize import word_tokenize
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
def filter_word_count(data, input_key='content', min_words=10, max_words=10000):
    assert isinstance(data, dict)
    # TODO: Implement word count filtering
    return data


@DataOperatorRegistry.register
def filter_colon_end(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement colon-ending line filtering
    return data


@DataOperatorRegistry.register
def filter_sentence_count(data, input_key='content', min_sentences=3, max_sentences=1000):
    assert isinstance(data, dict)
    # TODO: Implement sentence count filtering
    return data


@DataOperatorRegistry.register
def filter_ellipsis_end(data, input_key='content', max_ratio=0.3):
    assert isinstance(data, dict)
    # TODO: Implement ellipsis-ending line filtering
    return data


@DataOperatorRegistry.register
def filter_null_content(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement null/empty content filtering
    return data


@DataOperatorRegistry.register
def filter_word_length(data, input_key='content', min_length=3, max_length=15):
    assert isinstance(data, dict)
    # TODO: Implement mean word length filtering
    return data


@DataOperatorRegistry.register
def filter_symbol_ratio(data, input_key='content', max_ratio=0.3):
    assert isinstance(data, dict)
    # TODO: Implement symbol-to-word ratio filtering
    return data


@DataOperatorRegistry.register
def filter_idcard(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement ID card information filtering
    return data


@DataOperatorRegistry.register
def filter_no_punctuation(data, input_key='content', max_ratio=0.1):
    assert isinstance(data, dict)
    # TODO: Implement punctuation presence filtering
    return data


@DataOperatorRegistry.register
def filter_special_chars(data, input_key='content', max_ratio=0.25):
    assert isinstance(data, dict)
    # TODO: Implement special character ratio filtering
    return data


@DataOperatorRegistry.register
def filter_watermark(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement watermark detection and filtering
    return data


@DataOperatorRegistry.register
def filter_curly_brackets(data, input_key='content', max_ratio=0.2):
    assert isinstance(data, dict)
    # TODO: Implement curly bracket ratio filtering
    return data


@DataOperatorRegistry.register
def filter_capital_words(data, input_key='content', max_ratio=0.3):
    assert isinstance(data, dict)
    # TODO: Implement capital words ratio filtering
    return data


@DataOperatorRegistry.register
def filter_lorem_ipsum(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement Lorem Ipsum placeholder text detection
    return data


@DataOperatorRegistry.register
def filter_unique_words(data, input_key='content', min_ratio=0.3):
    assert isinstance(data, dict)
    # TODO: Implement unique words ratio filtering
    return data


@DataOperatorRegistry.register
def filter_char_count(data, input_key='content', min_chars=50, max_chars=100000):
    assert isinstance(data, dict)
    # TODO: Implement character count filtering
    return data


@DataOperatorRegistry.register
def filter_bullet_start(data, input_key='content', max_ratio=0.5):
    assert isinstance(data, dict)
    # TODO: Implement bullet point line filtering
    return data


@DataOperatorRegistry.register
def filter_javascript(data, input_key='content'):
    assert isinstance(data, dict)
    # TODO: Implement JavaScript code detection and filtering
    return data

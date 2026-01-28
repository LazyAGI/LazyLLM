import os
from ..base_data import DataOperatorRegistry
from lazyllm import LOG, config

import fasttext
from huggingface_hub import hf_hub_download
from datasketch import MinHash, MinHashLSH


@DataOperatorRegistry.register
class LanguageFilter:
    _model_cache = None

    def __init__(self, input_key='content', target_language='zh', model_cache_dir=None):
        self.input_key = input_key
        if isinstance(target_language, str):
            self.allowed_languages = [target_language]
        else:
            self.allowed_languages = target_language
        self.model_cache_dir = model_cache_dir or config['model_cache_dir']
        self.model = self._load_model()

    def _load_model(self):
        if LanguageFilter._model_cache is not None:
            return LanguageFilter._model_cache

        try:
            LOG.info('Downloading FastText language identification model from Hugging Face Hub...')
            model_path = hf_hub_download(
                repo_id='facebook/fasttext-language-identification',
                filename='model.bin',
                cache_dir=self.model_cache_dir
            )
            LanguageFilter._model_cache = fasttext.load_model(model_path)
            LOG.info('FastText language model loaded successfully.')
            return LanguageFilter._model_cache
        except Exception as e:
            LOG.error(f'Error downloading or loading FastText model: {e}')
            raise

    def __call__(self, data):
        assert isinstance(data, dict)

        if self.input_key not in data:
            return None

        text = data[self.input_key]
        if not isinstance(text, str) or not text.strip():
            return None

        labels, scores = self.model.predict(text.replace('\n', ' '), k=5)
        label_score_pairs = list(zip(labels, scores))
        label_score_pairs.sort(key=lambda x: x[1], reverse=True)

        top_labels = [label.replace('__label__', '') for label, _ in label_score_pairs]
        if any(label in self.allowed_languages for label in top_labels):
            return data

        return None


@DataOperatorRegistry.register
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
        assert isinstance(data, dict)

        if self.input_key not in data:
            return None

        text = data[self.input_key]
        if not isinstance(text, str) or not text.strip():
            return None

        minhash = self._create_minhash(text)
        result = self.lsh.query(minhash)

        if len(result) == 0:
            # New unique item, insert into LSH
            self.lsh.insert(self._item_counter, minhash)
            self._minhash_map[self._item_counter] = minhash
            self._item_counter += 1
            return data

        # Duplicate item, filter out
        return None


@DataOperatorRegistry.register
def filter_blocklist(data, input_key='content', blocklist=None):
    assert isinstance(data, dict)
    # TODO: Implement blocklist filtering
    return data


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

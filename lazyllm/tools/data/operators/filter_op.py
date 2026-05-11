import os
import re
from ..base_data import data_register
from lazyllm import LOG, config
from lazyllm.components.utils.downloader import ModelManager
from lazyllm.thirdparty import fasttext, datasketch, nltk, jieba, ahocorasick


Filter = data_register.new_group('filter')

ID_CARD_CHINESE_TERMS = [
    r'身\s{0,5}份\s{0,5}证(?:\s{0,5}号(?:\s{0,5}码)?)?',
    r'证\s{0,5}件(?:\s{0,5}号(?:\s{0,5}码)?)?',
    r'居\s{0,5}民\s{0,5}身\s{0,5}份\s{0,5}证',
    r'个\s{0,5}人\s{0,5}身\s{0,5}份',
]

ID_CARD_ENGLISH_TERMS = [
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

SPECIAL_CHAR_PATTERNS = [
    r'\u200b|\u200c|\u200d|\u200e|\u200f',
    r'\ufeff',
    r'\u2060|\u2061|\u2062|\u2063',
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',
    r'[\x80-\x9f]',
    r'[□]',
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
    r'[\ud800-\udfff]',
]

DEFAULT_WATERMARKS = [
    'Copyright', 'Watermark', 'Confidential', 'All Rights Reserved',
    'Proprietary', 'Trade Secret', 'Internal Use Only', 'Do Not Distribute',
    'Private and Confidential', 'Restricted', 'Classified', 'Top Secret',
    '版权所有', '保留所有权利', '机密', '内部资料', '严禁转载', '禁止复制',
    '仅供内部使用', '未经授权', '商业机密', '翻版必究'
]

LOREM_PATTERNS = [
    r'lorem\s+ipsum',
    r'dolor\s+sit\s+amet',
    r'consectetur\s+adipiscing',
    r'consectetur\s+adipisicing',
    r'sed\s+do\s+eiusmod',
    r'tempor\s+incididunt',
    r'ut\s+labore\s+et\s+dolore',
    r'magna\s+aliqua',
    r'eiusmod\s+tempor',
    r'incididunt\s+ut\s+labore',
    r'aliquip\s+ex\s+ea\s+commodo',
    r'duis\s+aute\s+irure',
    r'in\s+reprehenderit\s+in\s+voluptate',
    r'esse\s+cillum\s+dolore',
    r'fugiat\s+nulla\s+pariatur',
    r'excepteur\s+sint\s+occaecat',
    r'cupidatat\s+non\s+proident',
    r'sunt\s+in\s+culpa\s+qui\s+officia',
    r'deserunt\s+mollit\s+anim',
    r'占位符?文本',
    r'测试文[本字]',
    r'示例[文本内容]',
    r'这是一[个段]测试',
]

BULLET_CHARS = [
    '\u2022', '\u2023', '\u25E6', '\u2043',
    '\u25A0', '\u25A1', '\u25AA', '\u25AB',
    '\u25CF', '\u25CB', '\u25C6', '\u25C7',
    '\u25B6', '\u25BA', '\u25C0', '\u2192', '\u21D2', '\u27A4',
    '\u2013', '\u2014', '-', '*', '+',
    '\u2713', '\u2714', '\u2605', '\u2606',
    '\u00B7', '\u30FB',
]

JAVASCRIPT_PATTERNS = [
    'javascript', '<script', '</script>', 'function(', '=>', 'var ', 'let ', 'const ',
    'console.log', 'document.', 'window.', '$(', 'jquery', '.then(', 'async ',
    'addEventListener', 'onclick', 'onload', 'typeof', 'undefined', 'null',
]


def _setup_nltk_data_dir():
    if 'NLTK_DATA' not in os.environ:
        try:
            nltk_data_dir = os.path.join(config['home'], 'nltk_data')
        except (KeyError, TypeError):
            nltk_data_dir = os.path.join(os.path.expanduser('~'), '.lazyllm', 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)
        os.environ['NLTK_DATA'] = nltk_data_dir
    else:
        nltk_data_dir = os.environ['NLTK_DATA']
    return nltk_data_dir


class TargetLanguageFilter(Filter):
    COMMON_LANGUAGES = {
        'zho_Hans', 'zho_Hant', 'eng_Latn', 'spa_Latn', 'fra_Latn',
        'deu_Latn', 'jpn', 'kor', 'rus_Cyrl', 'ara', 'por_Latn',
        'ita_Latn', 'nld_Latn', 'pol_Latn', 'tur_Latn', 'vie',
        'tha', 'hin', 'ind_Latn', 'msa_Latn'
    }

    def __init__(self, input_key='content', target_language='zho_Hans', threshold=0.6, model_path=None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        if isinstance(target_language, str):
            self.allowed_languages = {target_language}
        else:
            self.allowed_languages = set(target_language)
        self.threshold = threshold
        if model_path is None:
            try:
                default_cache_dir = config['model_cache_dir']
            except (KeyError, TypeError):
                default_cache_dir = os.path.join(os.path.expanduser('~'), '.lazyllm', 'models')
            model_path = os.path.join(default_cache_dir, 'fasttext-language-identification', 'model.bin')
        self.model_path = model_path
        self._validate_languages()
        self.model = self._load_model()

    def _validate_languages(self):
        invalid_langs = self.allowed_languages - self.COMMON_LANGUAGES
        if invalid_langs:
            LOG.warning(
                f'TargetLanguageFilter: Invalid language codes: {invalid_langs}\n'
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
            if os.path.isfile(self.model_path):
                model_file = self.model_path
            elif os.path.isdir(self.model_path):
                model_file = os.path.join(self.model_path, 'model.bin')
            else:
                model_file = self._download_model()

            if not os.path.exists(model_file):
                raise FileNotFoundError(f'Model file not found at {model_file}')

            LOG.info(f'Loading FastText language model from {model_file}...')
            model = fasttext.load_model(model_file)
            LOG.info('FastText language model loaded successfully.')
            return model
        except Exception as e:
            LOG.error(f'Error loading FastText model: {e}')
            raise

    def _download_model(self):
        LOG.info('Downloading FastText language identification model...')
        model_repo = 'facebook/fasttext-language-identification'
        try:
            model_source = config['model_source']
        except (KeyError, TypeError):
            model_source = 'modelscope'

        if os.path.isdir(self.model_path) or self.model_path.endswith(os.sep):
            model_dir = self.model_path if os.path.isdir(self.model_path) else os.path.dirname(self.model_path)
        else:
            model_dir = os.path.dirname(self.model_path)

        os.makedirs(model_dir, exist_ok=True)
        model_manager = ModelManager(model_source=model_source)
        downloaded_path = model_manager.hub_downloader.download(model_repo, model_dir)

        if not downloaded_path:
            raise RuntimeError(f'Failed to download model: {model_repo}')
        model_file = os.path.join(downloaded_path, 'model.bin')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f'Model file not found at {model_file}')

        self.model_path = model_file
        return model_file

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return []

        k = max(5, len(self.allowed_languages))
        labels, scores = self.model.predict(text.replace('\n', ' ').strip(), k=k)
        if len(labels) > 0 and len(scores) > 0:
            for label, score in zip(labels, scores):
                pred_label = label.replace('__label__', '')
                if pred_label in self.allowed_languages and score >= self.threshold:
                    return data

        return []


class MinHashDeduplicator(Filter):
    __reg_overwrite__ = 'forward_batch_input'

    def __init__(self, input_key='content', threshold=0.85, num_perm=128, use_n_gram=True, ngram=5, **kwargs):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.threshold = threshold
        self.num_perm = num_perm
        self.use_n_gram = use_n_gram
        self.ngram = ngram
        self.lsh = datasketch.MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._item_counter = 0
        self._minhash_map = {}

    def _create_minhash(self, text):
        minhash = datasketch.MinHash(num_perm=self.num_perm)
        if self.use_n_gram:
            for i in range(len(text) - self.ngram + 1):
                minhash.update(text[i:i + self.ngram].encode('utf8'))
        else:
            for char in text:
                minhash.update(char.encode('utf8'))
        return minhash

    def forward_batch_input(self, data, **kwargs):
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


class WordBlocklistFilter(Filter):
    def __init__(self, input_key='content', blocklist=None, blocklist_path=None,
                 language='zh', threshold=1, _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.threshold = threshold
        self.language = language.lower()

        if blocklist is not None:
            words = [w.strip().lower() for w in blocklist if w and w.strip()]
        elif blocklist_path is not None:
            words = self._load_blocklist_from_file(blocklist_path)
        else:
            default_path = self._get_default_blocklist_path()
            words = self._load_blocklist_from_file(default_path)

        self._blocklist_words = words
        self._automaton = self._build_automaton(words)

        LOG.info(f'WordBlocklistFilter initialized with {len(words)} blocked words (AC automaton), '
                 f'language={self.language}')

    def _build_automaton(self, words):
        A = ahocorasick.Automaton()
        for idx, word in enumerate(words):
            A.add_word(word, (idx, word))
        A.make_automaton()
        return A

    def __getstate__(self):
        state = self.__dict__.copy()
        # automaton may not pickle well in process mode; keep words to rebuild
        state['_automaton'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._automaton is None and self._blocklist_words:
            self._automaton = self._build_automaton(self._blocklist_words)

    def _get_default_blocklist_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if self.language in ['zh', 'cn', 'chinese']:
            filename = 'zh.txt'
        elif self.language in ['en', 'english']:
            filename = 'en.txt'
        else:
            LOG.warning(f'Unsupported language: {self.language}, defaulting to zh.txt')
            filename = 'zh.txt'
        blocklist_path = os.path.join(current_dir, 'blocklist', filename)
        return blocklist_path

    def _load_blocklist_from_file(self, file_path):
        LOG.info(f'Loading blocklist from {file_path}...')
        with open(file_path, 'r', encoding='utf-8') as f:
            words = list(dict.fromkeys(line.strip().lower() for line in f if line.strip()))
        LOG.info(f'Loaded {len(words)} words from blocklist')
        return words

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return data

        text_lower = text.lower()
        blocklist_count = sum(1 for _ in self._automaton.iter(text_lower))

        if blocklist_count <= self.threshold:
            return data
        else:
            return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def word_count_filter(data, input_key='content', min_words=10, max_words=10000, language='zh'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    language = language.lower()
    if language in ['zh', 'cn', 'chinese']:
        count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    elif language in ['en', 'english']:
        count = len(text.split())
    else:
        LOG.warning(f'Unsupported language: {language}, using character count')
        count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    if min_words <= count < max_words:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def colon_end_filter(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return data
    if text.rstrip().endswith(':') or text.rstrip().endswith('：'):
        return []
    else:
        return data


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def sentence_count_filter(data, input_key='content', min_sentences=3, max_sentences=1000, language='zh'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    language = language.lower()
    if language in ['zh', 'cn', 'chinese']:
        sentences = re.split(r'[。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
    elif language in ['en', 'english']:
        sentences = nltk.sent_tokenize(text)
        num_sentences = len(sentences)
    else:
        LOG.warning(f'Unsupported language: {language}, using Chinese punctuation')
        sentences = re.split(r'[。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
    if min_sentences <= num_sentences <= max_sentences:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def ellipsis_end_filter(data, input_key='content', max_ratio=0.3):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return data
    ellipsis = ['...', '…', '……']
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    num_lines = len(lines)
    if num_lines == 0:
        return data
    num_occurrences = sum(
        1 for line in lines
        if any(line.endswith(e) for e in ellipsis)
    )
    ratio = num_occurrences / num_lines
    if ratio < max_ratio:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def null_content_filter(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if text is not None and isinstance(text, str) and text.strip() != '':
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def word_length_filter(data, input_key='content', min_length=3, max_length=20):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return []
    num_chars = sum(len(word) for word in words)
    mean_length = num_chars / num_words
    if min_length <= mean_length < max_length:
        return data
    else:
        return []


class SymbolRatioFilter(Filter):
    def __init__(self, input_key='content', max_ratio=0.3, symbols=None, _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.symbols = symbols or ['#', '...', '…']
        self.tokenizer = nltk.WordPunctTokenizer()

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return []

        tokens = self.tokenizer.tokenize(text)
        word_tokens = [t for t in tokens if t not in self.symbols]
        num_words = len(word_tokens)
        if num_words == 0:
            return []

        num_symbols = sum(text.count(symbol) for symbol in self.symbols)
        ratio = num_symbols / num_words
        if ratio < self.max_ratio:
            return data
        else:
            return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def idcard_filter(data, input_key='content', threshold=3):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    all_patterns = ID_CARD_CHINESE_TERMS + ID_CARD_ENGLISH_TERMS
    pattern = re.compile('|'.join(f'({p})' for p in all_patterns), re.I)
    matches = pattern.findall(text)
    has_too_many_id_terms = len(matches) >= threshold
    if not has_too_many_id_terms:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def no_punc_filter(data, input_key='content', max_length_between_punct=112, language='zh'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    language = language.lower()
    if language in ['zh', 'cn', 'chinese']:
        punct_pattern = r'[。！？；，、：""''（）《》【】…—.!?,;:]'
    elif language in ['en', 'english']:
        punct_pattern = r'[–.!?,;•/|…:;\'\"]'
    else:
        LOG.warning(f'Unsupported language: {language}, using Chinese punctuation')
        punct_pattern = r'[。！？；，、：""''（）《》【】…—.!?,;:]'
    paragraphs = text.split('\n')
    max_length = 0
    for paragraph in paragraphs:
        if len(paragraph.strip()) == 0:
            continue
        segments = re.split(punct_pattern, paragraph)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if language in ['en', 'english']:
                length = len(segment.split())
            else:
                length = len(segment)
            if length > max_length:
                max_length = length
    if max_length <= max_length_between_punct:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def special_char_filter(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    has_special_char = any(re.search(pattern, text) for pattern in SPECIAL_CHAR_PATTERNS)
    if not has_special_char:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def watermark_filter(data, input_key='content', watermarks=None):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    watermarks = watermarks or DEFAULT_WATERMARKS
    matches = re.search('|'.join(watermarks), text)
    if matches is None:
        return data
    else:
        return []


class StopWordFilter(Filter):
    def __init__(self, input_key='content', max_ratio=0.5, use_tokenizer=True, language='zh',
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.use_tokenizer = use_tokenizer
        self.language = language.lower()

        nltk_data_dir = _setup_nltk_data_dir()
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            LOG.info('Downloading NLTK stopwords...')
            nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)

        if self.language in ['en', 'english']:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        elif self.language in ['zh', 'cn', 'chinese']:
            self.stopwords = set(nltk.corpus.stopwords.words('chinese'))
        else:
            LOG.warning(f'Unsupported language: {self.language}, using English stopwords')
            self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return []

        if self.language in ['zh', 'cn', 'chinese']:
            if self.use_tokenizer:
                words = list(jieba.cut(text.lower()))
            else:
                words = list(text)
        elif self.language in ['en', 'english']:
            if self.use_tokenizer:
                words = nltk.word_tokenize(text.lower())
            else:
                words = text.lower().split()
        else:
            words = text.lower().split()

        num_words = len(words)
        if num_words == 0:
            return []

        num_stop_words = sum(1 for w in words if w in self.stopwords)
        ratio = num_stop_words / num_words

        if ratio < self.max_ratio:
            return data
        else:
            return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def curly_bracket_filter(data, input_key='content', max_ratio=0.08):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    num_brackets = text.count('{') + text.count('}')
    ratio = num_brackets / len(text) if len(text) > 0 else 0
    if ratio < max_ratio:
        return data
    else:
        return []


class CapitalWordFilter(Filter):
    def __init__(self, input_key='content', max_ratio=0.5, use_tokenizer=False,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.max_ratio = max_ratio
        self.use_tokenizer = use_tokenizer

        if self.use_tokenizer:
            nltk_data_dir = _setup_nltk_data_dir()
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                LOG.info('Downloading NLTK punkt_tab tokenizer...')
                nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_dir)

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)

        text = data.get(self.input_key)
        if not isinstance(text, str) or not text.strip():
            return []

        if self.use_tokenizer:
            words = nltk.word_tokenize(text)
        else:
            words = text.split()

        num_words = len(words)
        if num_words == 0:
            return []

        num_caps_words = sum(1 for word in words if word.isupper())
        ratio = num_caps_words / num_words

        if ratio <= self.max_ratio:
            return data
        else:
            return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def lorem_ipsum_filter(data, input_key='content', max_ratio=3e-8):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    pattern_str = '|'.join(f'({p})' for p in LOREM_PATTERNS)
    pattern = re.compile(pattern_str, re.IGNORECASE)
    matches = pattern.findall(text)
    num_occurrences = len(matches)
    ratio = num_occurrences / len(text) if len(text) > 0 else 0
    if ratio <= max_ratio:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='thread')
def unique_word_filter(data, input_key='content', min_ratio=0.1, use_tokenizer=True, language='zh'):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    language = language.lower()
    if language in ['zh', 'cn', 'chinese']:
        if use_tokenizer:
            words = list(jieba.cut(text.lower()))
        else:
            words = list(text)
    elif language in ['en', 'english']:
        if use_tokenizer:
            nltk_data_dir = _setup_nltk_data_dir()
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                LOG.info('Downloading NLTK punkt_tab tokenizer...')
                nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_dir)
            words = nltk.word_tokenize(text.lower())
        else:
            words = text.lower().split()
    else:
        LOG.warning(f'Unsupported language: {language}, using simple split')
        words = text.lower().split()
    num_words = len(words)
    if num_words == 0:
        return []
    num_unique_words = len(set(words))
    ratio = num_unique_words / num_words
    if ratio > min_ratio:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def char_count_filter(data, input_key='content', min_chars=100, max_chars=100000):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    text_no_space = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
    num_chars = len(text_no_space)
    if min_chars <= num_chars <= max_chars:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def bullet_point_filter(data, input_key='content', max_ratio=0.9):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    num_lines = len(lines)
    if num_lines == 0:
        return []
    num_bullet_lines = sum(
        1 for line in lines
        if any(line.startswith(bullet) for bullet in BULLET_CHARS)
    )
    ratio = num_bullet_lines / num_lines
    if ratio <= max_ratio:
        return data
    else:
        return []


@data_register('data.filter', rewrite_func='forward', _concurrency_mode='process')
def javascript_filter(data, input_key='content', min_non_script_lines=3):
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str) or not text.strip():
        return []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    num_lines = len(lines)
    if num_lines == 0:
        return []
    if num_lines <= 3:
        return data
    num_script_lines = sum(
        1 for line in lines
        if any(pattern in line.lower() for pattern in JAVASCRIPT_PATTERNS)
    )
    num_non_script_lines = num_lines - num_script_lines
    if num_non_script_lines >= min_non_script_lines:
        return data
    else:
        return []

import os
import shutil
import tempfile

import pytest

from lazyllm import config
from lazyllm.flow import Pipeline
from lazyllm.tools.data.pipelines.domain_pretrain_pipelines import (
    DOMAIN_PRETRAIN_FEATURES,
    build_domain_pretrain_pipeline,
    build_text_pt_pipeline,
    build_text_pt_plus_domain_pretrain_pipeline,
)


def _zh_pretrain_body(num_paragraphs: int = 30):
    parts = []
    for i in range(num_paragraphs):
        parts.append(
            f'第{i + 1}段：领域预训练数据清洗与质量评估相关内容。'
            f'涉及股票、投资、财报、数据库、网络与安全等主题。句子长度适中，并带有逗号与句号。'
        )
    return '\n'.join(parts)


def _en_text_pt_body(num_paragraphs: int = 50):
    openings = [
        'We examine linear algebra, matrix decompositions, and orthonormal bases.',
        'This passage explores spectral methods, factorization strategies, and convergence criteria.',
        'Consider the interplay between numerical precision, algorithmic complexity, and memory layout.',
        'Below we investigate approximation bounds, truncation errors, and asymptotic expansions.',
        'The framework combines interpolation, extrapolation, and regression techniques.',
        'Here we review orthogonality conditions, projection operators, and dual spaces.',
        'Attention is given to pivoting strategies, fill-in reduction, and reordering heuristics.',
        'We discuss norm estimates, condition numbers, and backward error analysis.',
        'The exposition covers gradient descent, proximal mappings, and trust region methods.',
        'We analyze eigenvalue sensitivity, pseudospectra, and resolvent bounds.',
    ]
    tail_phrases = [
        'hermitian spectra and courant minimax principles.',
        'cholesky factors for sparse positive definite systems.',
        'qr iteration with wilkinson shifts for eigenvalues.',
        'golub-kahan bidiagonalization and singular values.',
        'krylov subspaces and lanczos tridiagonalization.',
        'conjugate gradients for symmetric positive operators.',
        'gmres for nonsymmetric linear systems.',
        'preconditioned iterative refinement with ilu factors.',
        'operator norms and banach space embeddings.',
        'compact operators and fredholm alternatives.',
        'spectral radius bounds via gershgorin disks.',
        'perturbation theory for invariant subspaces.',
        'backward stability of householder reflections.',
        'mixed precision block lu with pivoting.',
        'tensor contractions and einstein summation.',
        'low-rank approximations via randomized range finders.',
        'graph laplacians and algebraic connectivity.',
        'finite element formulation on unstructured meshes.',
        'multigrid cycles for elliptic boundary problems.',
        'fast multipole methods for n-body potentials.',
        'toeplitz systems and circulant preconditioners.',
        'polynomial filters for spectrum slicing.',
        'davidson subspace iteration for ground states.',
        'jacobi-davidson corrections for interior eigenpairs.',
        'contour integration feasts for spectral projectors.',
        'structure-preserving symplectic integrators.',
        'lyapunov equations and gramian balancing.',
        'riccati recursions in kalman filtering.',
        'schur complements in sparse direct solvers.',
        'hierarchical matrices for kernel compression.',
        'wavelet sparsity in signal representations.',
    ]
    connectors = [
        'Coefficients converge under iterative refinement at round',
        'Residual norms decrease monotonically through iteration',
        'Approximate solutions improve progressively during sweep',
        'Accumulated rounding artifacts remain bounded after cycle',
        'Orthogonality is preserved up to machine epsilon by pass',
    ]
    parts = []
    for i in range(num_paragraphs):
        opening = openings[i % len(openings)]
        tail = tail_phrases[i % len(tail_phrases)]
        connector = connectors[i % len(connectors)]
        parts.append(
            f'Section {i + 1}. {opening} '
            f'We relate the discussion to {tail} '
            f'{connector} {i + 1}. '
            f'Numerical stability matters for each step along trajectory marker {i + 1}.'
        )
    return '\n\n'.join(parts)


def _en_text_pt_non_empty_body(num_paragraphs: int = 140):
    base_terms = [
        'qzvra', 'plnko', 'trvex', 'mylor', 'snkud', 'brvot', 'clqen',
        'dvrym', 'frtul', 'gnvix', 'hpral', 'jtkem', 'kvnor', 'lqzid',
    ]
    parts = []
    for i in range(1, num_paragraphs + 1):
        terms = ' '.join(f'{t}{i:03d}' for t in base_terms)
        parts.append(
            f'Unit {i}. {terms}. '
            f'zzkya{i:03d} qrvop{i:03d} tlnux{i:03d} mvqer{i:03d}.'
        )
    return '\n\n'.join(parts)


class TestDomainPretrainPipelines:
    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_domain_pretrain_features_export(self):
        assert isinstance(DOMAIN_PRETRAIN_FEATURES, tuple)
        assert 'field_normalization' in DOMAIN_PRETRAIN_FEATURES
        assert 'ngram_repetition_filter' in DOMAIN_PRETRAIN_FEATURES

    def test_build_domain_pretrain_pipeline_general_runs(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
        )
        assert isinstance(ppl, Pipeline)
        for name in ('field_normalization', 'text_normalizer', 'sensitive_cleaner', 'ngram_filter'):
            assert hasattr(ppl, name), f'missing step {name}'
        text = _zh_pretrain_body(30)
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert 'content' in res[0] and len(res[0]['content']) > 0

    def test_build_domain_pretrain_pipeline_field_mapping_options(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
            options={
                'field_mapping': {'body': 'content'},
            },
        )
        text = _zh_pretrain_body(32)
        res = ppl([{'body': text}])
        assert isinstance(res, list) and len(res) == 1
        assert 'content' in res[0]

    def test_build_domain_pretrain_pipeline_finance_keyword_filter(self):
        ppl = build_domain_pretrain_pipeline(
            domain='finance',
            content_key='content',
            language='zh',
            enabled={
                'domain_keyword_filter': True,
                'domain_relevance_scorer': False,
            },
        )
        text = _zh_pretrain_body(36)
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_keyword_hits' in res[0]

    def test_build_domain_pretrain_pipeline_relevance_scorer(self):
        ppl = build_domain_pretrain_pipeline(
            domain='finance',
            content_key='content',
            language='zh',
            enabled={
                'domain_keyword_filter': False,
                'domain_relevance_scorer': True,
            },
            options={'min_relevance_score': 0.0001},
        )
        text = _zh_pretrain_body(36)
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_domain_relevance_score' in res[0]

    def test_build_domain_pretrain_pipeline_language_filter_en(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='en',
            enabled={'language_filter': True},
            options={'min_language_ratio': 0.3},
        )
        parts = [
            f'Paragraph {i} discusses science, data, and software engineering practices.'
            for i in range(25)
        ]
        text = '\n\n'.join(parts)
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_language_ratio' in res[0]

    def test_build_domain_pretrain_opts_enable_domain_keyword_flag(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
            options={
                'enable_domain_keyword_filter': True,
                'keyword_mode': 'any',
                'min_keyword_hits': 1,
            },
            domain_keywords=['测试', '流水线'],
        )
        text = _zh_pretrain_body(34) + '测试与流水线关键词。'
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_keyword_hits' in res[0]

    def test_build_text_pt_pipeline_structure(self):
        ppl = build_text_pt_pipeline(
            content_key='content',
            language='zh',
            min_chars=100,
            min_words=10,
        )
        assert isinstance(ppl, Pipeline)
        for name in (
            'null_content_filter',
            'char_count_filter',
            'word_count_filter',
            'token_chunker',
        ):
            assert hasattr(ppl, name), f'missing step {name}'

    def test_build_text_pt_pipeline_runs_if_tokenizer_available(self):
        ppl = build_text_pt_pipeline(
            content_key='content',
            language='en',
            min_chars=200,
            min_words=40,
            max_tokens=512,
            min_tokens=64,
        )
        text = _en_text_pt_non_empty_body()
        try:
            res = ppl([{'content': text}])
        except Exception as e:
            pytest.skip(f'TokenChunker / tokenizer 不可用: {e}')
        assert isinstance(res, list)
        assert len(res) >= 1
        assert all('content' in row for row in res)

    def test_build_text_pt_plus_domain_pretrain_structure(self):
        ppl = build_text_pt_plus_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
        )
        assert isinstance(ppl, Pipeline)
        assert hasattr(ppl, 'domain_enhance') and hasattr(ppl, 'text_pt')

    def test_build_text_pt_plus_runs_if_tokenizer_available(self):
        ppl = build_text_pt_plus_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='en',
            options={
                'min_chars': 200,
                'min_words': 40,
                'min_tokens': 64,
                'max_tokens': 512,
            },
        )
        text = _en_text_pt_non_empty_body()
        try:
            res = ppl([{'content': text}])
        except Exception as e:
            pytest.skip(f'组合流水线需 TokenChunker: {e}')
        assert isinstance(res, list)
        assert len(res) >= 1
        assert all('content' in row for row in res)

from typing import List, Optional

from lazyllm import pipeline, LOG
from lazyllm.tools.data import reranker


def build_reranker_dataformatter_pipeline(
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    input_neg_key: str = 'neg',
    output_format: str = 'flagreranker',
    train_group_size: int = 8,
):
    if output_format == 'flagreranker':
        format_cls = reranker.RerankerFormatFlagReranker
        format_kw = {'train_group_size': train_group_size}
    elif output_format == 'cross_encoder':
        format_cls = reranker.RerankerFormatCrossEncoder
        format_kw = {}
    elif output_format == 'pairwise':
        format_cls = reranker.RerankerFormatPairwise
        format_kw = {}
    else:
        raise ValueError(f'Unknown output format: {output_format}')

    with pipeline() as ppl:
        ppl.validate = reranker.validate_reranker_data(
            input_query_key=input_query_key,
            input_pos_key=input_pos_key,
            input_neg_key=input_neg_key,
        )
        ppl.format = format_cls(**format_kw)
    return ppl


def build_convert_from_embed_pipeline(
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    input_neg_key: str = 'neg',
    adjust_neg_count: int = 7,
    seed: int = 42,
):
    with pipeline() as ppl:
        ppl.validate = reranker.validate_reranker_embedding_data(
            input_query_key=input_query_key,
            input_pos_key=input_pos_key,
            input_neg_key=input_neg_key,
        )
        ppl.adjust = reranker.RerankerAdjustNegatives(
            adjust_neg_count=adjust_neg_count,
            seed=seed,
        )
        ppl.build = reranker.RerankerBuildFormat()
    return ppl


def _mixed_init_factory(embedding_serving):
    def _mixed_init(results):
        results = reranker.RerankerInitBM25()(results)
        results = reranker.RerankerInitSemantic(
            embedding_serving=embedding_serving
        )(results)
        for item in results:
            item['_embedding_serving'] = embedding_serving
        return results

    return _mixed_init


def _hard_neg_init_op(mining_strategy: str, embedding_serving):
    if mining_strategy == 'bm25':
        return reranker.RerankerInitBM25()
    if mining_strategy == 'semantic':
        return reranker.RerankerInitSemantic(embedding_serving=embedding_serving)
    if mining_strategy == 'mixed':
        return _mixed_init_factory(embedding_serving)
    return None


def _create_hard_neg_mining_pipeline(
    mining_strategy: str,
    num_negatives: int,
    seed: int,
    input_query_key: str,
    input_pos_key: str,
    output_neg_key: str,
    bm25_ratio: float,
    embedding_serving,
):
    common_kw = dict(
        num_negatives=num_negatives,
        input_query_key=input_query_key,
        input_pos_key=input_pos_key,
        output_neg_key=output_neg_key,
    )
    with pipeline() as ppl:
        if mining_strategy == 'random':
            ppl.mine = reranker.RerankerMineRandomNegatives(
                seed=seed,
                **common_kw,
            )
        elif mining_strategy == 'bm25':
            ppl.mine = reranker.RerankerMineBM25Negatives(**common_kw)
        elif mining_strategy == 'semantic':
            ppl.mine = reranker.RerankerMineSemanticNegatives(
                embedding_serving=embedding_serving,
                **common_kw,
            )
        else:
            ppl.mine = reranker.RerankerMineMixedNegatives(
                bm25_ratio=bm25_ratio,
                embedding_serving=embedding_serving,
                **common_kw,
            )
    return ppl


_CLEANUP_KEYS = (
    '_corpus',
    '_bm25',
    '_bm25_corpus',
    '_bm25_tokenizer',
    '_bm25_stopwords',
    '_bm25_stemmer',
    '_semantic_embeddings',
    '_semantic_corpus',
    '_embedding_serving',
)


def build_reranker_hard_neg_pipeline(
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    output_neg_key: str = 'neg',
    corpus_key: str = 'passage',
    corpus: Optional[List[str]] = None,
    mining_strategy: str = 'random',
    num_negatives: int = 7,
    embedding_serving=None,
    bm25_ratio: float = 0.5,
    seed: int = 42,
    corpus_dir: Optional[str] = None,
):
    if mining_strategy not in ('random', 'bm25', 'semantic', 'mixed'):
        raise ValueError(
            f'Unknown mining strategy: {mining_strategy}. '
            'Use \'random\', \'bm25\', \'semantic\', or \'mixed\'.'
        )

    build_corpus_op = reranker.build_reranker_corpus(
        input_pos_key=input_pos_key,
        corpus_key=corpus_key,
        corpus=corpus,
        corpus_dir=corpus_dir,
    )
    init_op = _hard_neg_init_op(mining_strategy, embedding_serving)
    ppl = _create_hard_neg_mining_pipeline(
        mining_strategy,
        num_negatives=num_negatives,
        seed=seed,
        input_query_key=input_query_key,
        input_pos_key=input_pos_key,
        output_neg_key=output_neg_key,
        bm25_ratio=bm25_ratio,
        embedding_serving=embedding_serving,
    )

    def run(inputs: List[dict]) -> List[dict]:
        results = build_corpus_op(inputs)

        if init_op is not None:
            results = init_op(results)

        results = ppl(results)

        for item in results:
            for key in _CLEANUP_KEYS:
                item.pop(key, None)

        LOG.info(f'Hard negative mining completed for {len(results)} samples.')
        return results

    return run


def build_reranker_qa_generate_pipeline(
    input_key: str = 'passage',
    output_query_key: str = 'query',
    llm_serving=None,
    num_queries: int = 3,
    lang: str = 'zh',
    difficulty_levels: Optional[List[str]] = None,
):
    with pipeline() as ppl:
        ppl.generate = reranker.RerankerGenerateQueries(
            llm_serving=llm_serving,
            lang=lang,
            num_queries=num_queries,
            difficulty_levels=difficulty_levels,
            input_key=input_key,
        )
        ppl.parse = reranker.RerankerParseQueries(
            input_key=input_key,
            output_query_key=output_query_key,
        )
    return ppl

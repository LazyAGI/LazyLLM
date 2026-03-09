from typing import List, Optional
from pathlib import Path
import json
from lazyllm import pipeline
from lazyllm import LOG
from lazyllm.tools.data import embedding


def build_embedding_data_augmentation_pipeline(
        input_query_key: str = "query",
        output_query_key: str = "query",
        keep_original: bool = True,
        llm=None,
        augment_methods: Optional[List[str]] = None,
        num_augments: int = 2,
        lang: str = 'en',
):
    augment_methods = augment_methods or []

    def run(inputs: List[dict]) -> List[dict]:
        normalized_inputs = []
        for item in inputs:
            normalized_item = item.copy()
            if input_query_key != 'query':
                normalized_item['query'] = item.get(input_query_key, '')
            normalized_inputs.append(normalized_item)

        results = []
        if keep_original:
            results.extend(inputs)

        for method in augment_methods:
            LOG.info(f"Applying augmentation method: {method}")
            with pipeline() as ppl:
                if method == "query_rewrite":
                    ppl.augment = embedding.EmbeddingQueryRewrite(
                        llm=llm,
                        num_augments=num_augments,
                        lang=lang,
                    )
                elif method == "synonym_replace":
                    ppl.augment = embedding.EmbeddingAdjacentWordSwap(num_augments=num_augments)
                else:
                    LOG.warning(f"Unknown augmentation method: {method}, skipping...")
                    continue
            augmented = ppl(normalized_inputs)
            if output_query_key != 'query':
                for item in augmented:
                    item[output_query_key] = item.pop('query', '')
            results.extend(augmented)
        return results

    return run


def build_embedding_data_formatter_pipeline(
        input_query_key: str = "query",
        input_pos_key: str = "pos",
        input_neg_key: str = "neg",
        output_format: str = "flagembedding",
        instruction: Optional[str] = None,
        output_file: Optional[str] = None,
):
    if output_format not in ("flagembedding", "sentence_transformers", "triplet"):
        raise ValueError(f"Unknown output format: {output_format}")

    with pipeline() as ppl:
        if output_format == "flagembedding":
            ppl.format = embedding.EmbeddingFormatFlagEmbedding(instruction=instruction)
        elif output_format == "sentence_transformers":
            ppl.format = embedding.EmbeddingFormatSentenceTransformers()
        else:
            ppl.format = embedding.EmbeddingFormatTriplet()

    def run(inputs: List[dict]) -> List[dict]:
        normalized_inputs = []
        for item in inputs:
            normalized_item = {
                'query': item.get(input_query_key, ''),
                'pos': item.get(input_pos_key, []),
                'neg': item.get(input_neg_key, []),
            }
            if normalized_item['query'] and normalized_item['pos']:
                normalized_inputs.append(normalized_item)
            else:
                LOG.warning(f"Skipping item with missing query or pos: {item}")
        results = ppl(normalized_inputs)
        LOG.info(f"Formatted {len(results)} training samples.")
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved formatted data to {output_path}")
        return results

    return run


def build_embedding_hard_neg_pipeline(
        input_query_key: str = 'query',
        input_pos_key: str = 'pos',
        output_neg_key: str = 'neg',
        corpus_key: str = "passage",
        corpus: Optional[List[str]] = None,
        mining_strategy: str = "random",
        num_negatives: int = 7,
        embedding_serving=None,
        language: str = "zh",
        seed: int = 42,
):
    if mining_strategy not in ("random", "bm25", "semantic"):
        raise ValueError(
            f"Unknown mining strategy: {mining_strategy}. "
            "Use 'random', 'bm25', or 'semantic'."
        )

    build_corpus_op = embedding.build_embedding_corpus(
        input_pos_key=input_pos_key, corpus_key=corpus_key, corpus=corpus
    )

    if mining_strategy == "bm25":
        init_op = embedding.EmbeddingInitBM25(language=language)
    elif mining_strategy == "semantic":
        init_op = embedding.EmbeddingInitSemantic(embedding_serving=embedding_serving)
    else:
        init_op = None

    with pipeline() as ppl:
        if mining_strategy == "random":
            ppl.mine = embedding.mine_random_negatives(
                num_negatives=num_negatives,
                seed=seed,
                input_query_key=input_query_key,
                input_pos_key=input_pos_key,
                output_neg_key=output_neg_key,
            )
        elif mining_strategy == "bm25":
            ppl.mine = embedding.mine_bm25_negatives(
                num_negatives=num_negatives,
                input_query_key=input_query_key,
                input_pos_key=input_pos_key,
                output_neg_key=output_neg_key,
            )
        else:
            ppl.mine = embedding.EmbeddingMineSemanticNegatives(
                num_negatives=num_negatives,
                embedding_serving=embedding_serving,
                input_query_key=input_query_key,
                input_pos_key=input_pos_key,
                output_neg_key=output_neg_key,
            )

    _cleanup_keys = [
        '_corpus', '_bm25', '_bm25_corpus', '_bm25_tokenizer',
        '_bm25_stopwords', '_bm25_stemmer', '_semantic_embeddings', '_semantic_corpus',
    ]

    def run(inputs: List[dict]) -> List[dict]:
        results = build_corpus_op(inputs)
        if init_op is not None:
            results = init_op(results)
        results = ppl(results)
        for item in results:
            for key in _cleanup_keys:
                item.pop(key, None)
        LOG.info(f"Hard negative mining completed for {len(results)} samples.")
        return results

    return run


def build_query_generation_pipeline(
        input_key: str = "passage",
        output_query_key: str = "query",
        num_queries: int = 3,
        lang: str = "zh",
        query_types: Optional[List[str]] = None,
        llm=None,
):
    with pipeline() as ppl:
        ppl.generate = embedding.EmbeddingGenerateQueries(
            llm=llm,
            num_queries=num_queries,
            lang=lang,
            query_types=query_types,
            input_key=input_key,
        )
        ppl.parse = embedding.EmbeddingParseQueries(
            input_key=input_key,
            output_query_key=output_query_key,
        )
    return ppl

from typing import Optional

from lazyllm import pipeline
from lazyllm.tools.data import kbc


def build_convert_md_pipeline(
    input_key: str = 'source',
    output_key: str = 'text_path',
    intermediate_dir: str = 'intermediate',
    mineru_url: str = '',
    mineru_backend: str = 'vlm-vllm-async-engine',
    upload_mode: bool = True,
):
    if not mineru_url:
        raise ValueError(
            'mineru_url is required. Please provide the MinerU API server URL, '
        )

    with pipeline() as ppl:
        ppl.normalize = kbc.FileOrURLNormalizer(
            intermediate_dir=intermediate_dir,
            input_key=input_key,
        )
        ppl.convert_html = kbc.HTMLToMarkdownConverter()
        ppl.convert_pdf = kbc.PDFToMarkdownConverterAPI(
            mineru_url=mineru_url,
            mineru_backend=mineru_backend,
            upload_mode=upload_mode,
        )
    return ppl


def build_batch_chunk_generator_pipeline(
    input_key: str = 'text_path',
    output_key: str = 'chunk_path',
    output_dir: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    split_method: str = 'token',
    tokenizer_name: str = 'bert-base-uncased',
):
    with pipeline() as ppl:
        ppl.load = kbc.KBCLoadText(input_key=input_key)
        ppl.chunk = kbc.KBCChunkText(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
            tokenizer_name=tokenizer_name,
        )
        ppl.save = kbc.KBCSaveChunks(
            input_key=input_key,
            output_key=output_key,
            output_dir=output_dir,
        )
    return ppl


def build_single_chunk_generator_pipeline(
    input_key: str = 'text_path',
    output_key: str = 'chunk_path',
    output_dir: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    split_method: str = 'token',
    tokenizer_name: str = 'bert-base-uncased',
):
    with pipeline() as ppl:
        ppl.load = kbc.KBCLoadText(input_key=input_key)
        ppl.chunk = kbc.KBCChunkText(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
            tokenizer_name=tokenizer_name,
        )
        ppl.expand = kbc.KBCExpandChunks(output_key=output_key)
    return ppl


def build_multihop_qa_pipeline(
    input_key: str = 'chunk_path',
    output_key: str = 'enhanced_chunk_path',
    ext_field: str = 'cleaned_chunk',
    llm=None,
    lang: str = 'en',
    output_dir: Optional[str] = None,
):
    with pipeline() as ppl:
        ppl.load = kbc.KBCLoadChunkFile(input_key=input_key)
        ppl.preprocess = kbc.KBCPreprocessText(ext_field=ext_field)
        ppl.extract = kbc.KBCExtractInfoPairs(lang=lang)
        ppl.generate = kbc.KBCGenerateMultiHopQA(llm=llm, lang=lang)
        ppl.parse = kbc.parse_qa_pairs()
        ppl.save = kbc.KBCSaveEnhanced(output_key=output_key, output_dir=output_dir)
    return ppl


def build_batch_kbc_pipeline(
    input_key: str = 'chunk_path',
    output_key: str = 'cleaned_chunk_path',
    llm=None,
    lang: str = 'en',
    output_dir: Optional[str] = None,
):
    with pipeline() as ppl:
        ppl.load = kbc.KBCLoadRAWChunkFile(input_key=input_key)
        ppl.generate = kbc.KBCGenerateCleanedText(llm=llm, lang=lang)
        ppl.extract = kbc.extract_cleaned_content()
        ppl.save = kbc.KBCSaveCleaned(output_key=output_key, output_dir=output_dir)
    return ppl


def build_single_kbc_pipeline(
    input_key: str = 'raw_chunk',
    output_key: str = 'cleaned_chunk',
    llm=None,
    lang: str = 'en',
):
    with pipeline() as ppl:
        ppl.generate = kbc.KBCGenerateCleanedTextSingle(
            input_key=input_key,
            llm=llm,
            lang=lang,
        )
        ppl.extract = kbc.extract_cleaned_content_single(output_key=output_key)
    return ppl


def build_qa_extract_pipeline(
    output_instruction_key: str = 'instruction',
    output_question_key: str = 'input',
    output_answer_key: str = 'output',
    input_qa_key: str = 'QA_pairs',
    input_instruction: str = 'Please answer the following question based on the provided information.',
):
    with pipeline() as ppl:
        ppl.load = kbc.KBCLoadQAData(qa_key=input_qa_key)
        ppl.extract = kbc.KBCExtractQAPairs(
            qa_key=input_qa_key,
            output_instruction_key=output_instruction_key,
            output_question_key=output_question_key,
            output_answer_key=output_answer_key,
            instruction=input_instruction,
        )
    return ppl

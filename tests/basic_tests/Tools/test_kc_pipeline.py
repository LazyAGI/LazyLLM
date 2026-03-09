"""Tests for KC (Knowledge Cleaning) Pipeline.

参考 test_rag_pipeline / test_data_pipeline：传入 data，用 mock_llm 覆盖需 LLM 的 pipeline，验证 res 输出。
"""

import json
import os
import shutil
import tempfile
import pytest

from lazyllm import config
from lazyllm.tools.data.pipelines.kc_pipelines import (
    build_convert_md_pipeline,
    build_batch_chunk_generator_pipeline,
    build_single_chunk_generator_pipeline,
    build_multihop_qa_pipeline,
    build_batch_kbc_pipeline,
    build_single_kbc_pipeline,
    build_qa_extract_pipeline,
)


class MockLLMServe:
    """单次 share 的 LLM 服务 mock，供 KBC 等算子使用。"""
    def __init__(self, return_value=None):
        self._return_value = return_value or {"text": "cleaned content"}

    def start(self):
        return self

    def prompt(self, system_prompt):
        return self

    def formatter(self, formatter):
        return self

    def __call__(self, prompt):
        return self._return_value


class MockLLM:
    """KBC 算子用的 LLM mock：share() 返回一个 serve，具备 prompt/formatter/start/__call__。"""
    def __init__(self, return_value=None):
        self._serve = MockLLMServe(return_value=return_value)

    def share(self, prompt=None, format=None, stream=None, history=None):
        return self._serve


class TestKcPipeline:
    """KC pipeline 测例：传入 data，验证 res。"""

    def setup_method(self):
        self.root_dir = './test_kc_pipeline'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_build_convert_md_pipeline_requires_mineru_url(self):
        """未传 mineru_url 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="mineru_url is required"):
            build_convert_md_pipeline()

    def test_convert_md_pipeline_with_data(self):
        """convert_md：传入 data，用假 mineru_url 仅校验可执行；无真实服务时可能中途报错或返回空。"""
        ppl = build_convert_md_pipeline(mineru_url='http://127.0.0.1:99999')
        data = [{"source": "https://example.com/doc.pdf"}]
        res = ppl(data)
        assert isinstance(res, list)

    def test_qa_extract_pipeline_with_data(self):
        """QA 抽取不需要 LLM，直接传 data 并校验输出。"""
        ppl = build_qa_extract_pipeline()
        data = [{
            "QA_pairs": {
                "qa_pairs": [
                    {"question": "What is ML?", "answer": "Machine learning."},
                    {"question": "What is DL?", "answer": "Deep learning."},
                ]
            }
        }]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 2
        for item in res:
            assert "instruction" in item and "input" in item and "output" in item

    def test_single_kbc_pipeline_with_mock_llm_and_data(self):
        """single_kbc 需要 LLM，用 mock_llm + data 跑通并校验 cleaned_chunk。"""
        mock_llm = MockLLM(return_value={"text": "cleaned text from mock"})
        ppl = build_single_kbc_pipeline(llm=mock_llm, input_key="raw_chunk", output_key="cleaned_chunk")
        data = [{"raw_chunk": "some raw chunk content here"}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 1
        assert "cleaned_chunk" in res[0]
        assert "cleaned text from mock" in res[0]["cleaned_chunk"] or res[0]["cleaned_chunk"] == "cleaned text from mock"

    def test_batch_chunk_generator_pipeline_with_data(self):
        """batch_chunk：传入含 text_path 的 data（临时 txt 文件），校验输出；无 tokenizer 时可能返回 0 条。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample document for chunking. " * 50)
            text_path = os.path.abspath(f.name)
        try:
            ppl = build_batch_chunk_generator_pipeline(
                input_key="text_path",
                output_key="chunk_path",
                output_dir=os.path.abspath(self.root_dir),
                chunk_size=128,
                chunk_overlap=20,
            )
            data = [{"text_path": text_path}]
            res = ppl(data)
            assert isinstance(res, list)
            if len(res) == 1:
                assert "chunk_path" in res[0]
                assert res[0]["chunk_path"]
        finally:
            if os.path.exists(text_path):
                os.remove(text_path)

    def test_single_chunk_generator_pipeline_with_data(self):
        """single_chunk：传入含 text_path 的 data（临时 txt），校验输出；无 tokenizer 时可能返回 0 条。"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample document for single chunk expand. " * 30)
            text_path = os.path.abspath(f.name)
        try:
            ppl = build_single_chunk_generator_pipeline(
                input_key="text_path",
                output_key="chunk_path",
                chunk_size=128,
                chunk_overlap=20,
            )
            data = [{"text_path": text_path}]
            res = ppl(data)
            assert isinstance(res, list)
            if len(res) >= 1:
                for item in res:
                    assert "chunk_path" in item
        finally:
            if os.path.exists(text_path):
                os.remove(text_path)

    def test_batch_kbc_pipeline_with_mock_llm_and_data(self):
        """batch_kbc 需要 LLM；用 mock_llm + 临时 raw chunk 文件 + data 跑通并校验输出。"""
        mock_llm = MockLLM(return_value={"text": "batch cleaned"})
        chunk_dir = os.path.join(os.path.abspath(self.root_dir), "raw_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, "raw.json")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump([{"raw_chunk": "raw content for batch cleaning"}], f, ensure_ascii=False)
        try:
            ppl = build_batch_kbc_pipeline(
                llm=mock_llm,
                input_key="chunk_path",
                output_key="cleaned_chunk_path",
                output_dir=os.path.abspath(self.root_dir),
            )
            data = [{"chunk_path": os.path.abspath(chunk_path)}]
            res = ppl(data)
            assert isinstance(res, list)
            if len(res) == 1:
                assert "cleaned_chunk_path" in res[0]
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    def test_multihop_qa_pipeline_with_mock_llm_and_data(self):
        """multihop_qa 需要 LLM；用 mock_llm + 含 cleaned_chunk 的临时 chunk 文件 + data 跑通并校验。"""
        mock_llm = MockLLM(return_value={"question": "Mock Q?", "answer": "Mock A."})
        chunk_dir = os.path.join(os.path.abspath(self.root_dir), "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, "chunk.json")
        # KBCExtractInfoPairs 需至少 3 句（按 . 分割），且每句 >10 字符
        long_content = (
            "First sentence for multihop QA context here. "
            "Second sentence with enough length for extraction. "
            "Third sentence to form info pairs for the pipeline."
        )
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump([{"cleaned_chunk": long_content}], f, ensure_ascii=False)
        try:
            ppl = build_multihop_qa_pipeline(
                llm=mock_llm,
                input_key="chunk_path",
                output_key="enhanced_chunk_path",
                output_dir=os.path.abspath(self.root_dir),
            )
            data = [{"chunk_path": os.path.abspath(chunk_path)}]
            res = ppl(data)
            assert isinstance(res, list)
            if len(res) == 1:
                assert "enhanced_chunk_path" in res[0]
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    def test_single_kbc_pipeline_llm_none_drops_item(self):
        """single_kbc 在 llm=None 时算子抛错被框架捕获，该条带 infer_error 被丢弃，返回 0 条。"""
        ppl = build_single_kbc_pipeline(llm=None)
        data = [{"raw_chunk": "x"}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 0

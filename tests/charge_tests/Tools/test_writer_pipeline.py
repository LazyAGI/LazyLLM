from typing import List

import pytest
from pydantic import BaseModel

import lazyllm
from lazyllm.tools.writer.tools.base import WriterToolBase
from ...utils import get_api_key, get_path


BASE_PATH = "lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py"
WRITER_BASE_PATH = "lazyllm/tools/writer/tools/base.py"
QWEN_MODEL = "qwen-turbo"


class WriterStructuredProbe(BaseModel):
    title: str
    section_count: int
    keywords: List[str]


@pytest.mark.ignore_cache_on_change(BASE_PATH, get_path("qwen"), WRITER_BASE_PATH)
def test_writer_call_llm_structured_with_qwen():
    llm = lazyllm.OnlineChatModule(
        source="qwen",
        model=QWEN_MODEL,
        api_key=get_api_key("qwen"),
        stream=False,
    )
    tool = WriterToolBase(llm=llm)

    result = tool._call_llm_structured(
        (
            "Generate a compact JSON object for testing WriterToolBase structured LLM output. "
            "Use title 'Writer Pipeline Structured Output Test', section_count 3, "
            "and include the keywords planning, drafting, and review."
        ),
        WriterStructuredProbe,
    )

    assert isinstance(result, WriterStructuredProbe)
    assert result.title
    assert result.section_count == 3
    assert {"planning", "drafting", "review"}.issubset(set(result.keywords))

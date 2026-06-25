import os

from lazyllm.tools.writer.data_models import (
    AuditResult,
    DraftBlock,
    DraftDocument,
    DraftSection,
    ResourceProfile,
    ReviewReport,
    SectionInstruction,
    WritingOutput,
    WritingOutline,
    OutlineNode,
)
from lazyllm.tools.writer.tools.base import WriterToolBase
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.utils import load_artifact_json
from lazyllm.tools.writer.workflow.naive_writer_workflow import NaiveWriterWorkflow


class RecordingResourceTools(WriterToolBase):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def profile_resources(self, task, input_resources=None):
        self.calls.append(("profile_resources", task, input_resources))
        profiles = [ResourceProfile(resource_id="res-1", resource_role="background", summary="背景资料")]
        return self._save_artifacts(
            {"resource_profiles": profiles},
            step_name="profile_resources",
            primary_key="resource_profiles",
            context_key=None,
            summary="Profiled resources.",
            counts={"resource_profiles": 1},
        ).model_dump()


class RecordingContextTools(WriterContextTools):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def create_writing_context(self, task, resource_profiles=None, doc_ir=None):
        self.calls.append(("create_writing_context", task, resource_profiles, doc_ir))
        return super().create_writing_context(task=task, resource_profiles=resource_profiles, doc_ir=doc_ir)

    def update_writing_context(self, content_artifact, context):
        self.calls.append(("update_writing_context", content_artifact, context))
        return super().update_writing_context(content_artifact=content_artifact, context=context)


class RecordingPlanningTools(WriterToolBase):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def generate_outline(self, task, context, resource_profiles=None, execution_results=None):
        self.calls.append(("generate_outline", task, context, resource_profiles, execution_results))
        outline = WritingOutline(title="方案", nodes=[OutlineNode(node_id="n1", title="背景")])
        return self._save_artifacts(
            {"outline": outline},
            step_name="generate_outline",
            primary_key="outline",
            context_key=None,
            summary="Generated outline.",
            counts={"outline_nodes": 1},
        ).model_dump()

    def generate_section_instructions(self, outline, context, execution_results=None):
        self.calls.append(("generate_section_instructions", outline, context, execution_results))
        instructions = [
            SectionInstruction(
                instruction_id="ins-1",
                outline_node_id="n1",
                section_title="背景",
                section_goal="介绍背景",
            )
        ]
        return self._save_artifacts(
            {"section_instructions": instructions},
            step_name="generate_section_instructions",
            primary_key="section_instructions",
            context_key=None,
            summary="Generated section instructions.",
            counts={"section_instructions": 1},
        ).model_dump()


class RecordingDraftingTools(WriterToolBase):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def generate_draft_section(self, task, section_instruction, context, previous_sections=None):
        self.calls.append(("generate_draft_section", task, section_instruction, context, previous_sections))
        section = DraftSection(
            section_id="sec-1",
            outline_node_id="n1",
            title="背景",
            blocks=[DraftBlock(block_id="b1", section_id="sec-1", content="背景内容")],
        )
        return self._save_artifacts(
            {"draft_section": section},
            step_name="generate_draft_section",
            primary_key="draft_section",
            context_key=None,
            summary="Generated draft section.",
            counts={"draft_blocks": 1},
        ).model_dump()

    def generate_draft_document(self, draft_sections, context):
        self.calls.append(("generate_draft_document", draft_sections, context))
        document = DraftDocument(
            draft_id="draft-1",
            title="方案",
            sections=[
                DraftSection(
                    section_id="sec-1",
                    outline_node_id="n1",
                    title="背景",
                    blocks=[DraftBlock(block_id="b1", section_id="sec-1", content="背景内容")],
                )
            ],
        )
        return self._save_artifacts(
            {"draft_document": document},
            step_name="generate_draft_document",
            primary_key="draft_document",
            context_key=None,
            summary="Generated draft document.",
            counts={"draft_sections": 1},
        ).model_dump()

    def generate_writing_output(self, draft, context):
        self.calls.append(("generate_writing_output", draft, context))
        output = WritingOutput(title="方案", content="# 方案\n\n背景内容")
        return self._save_artifacts(
            {"writing_output": output},
            step_name="generate_writing_output",
            primary_key="writing_output",
            context_key=None,
            summary="Generated writing output.",
            counts={"characters": len(output.content)},
        ).model_dump()


class RecordingQualityTools(WriterToolBase):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def validate_section(self, draft_section, section_instruction, context):
        self.calls.append(("validate_section", draft_section, section_instruction, context))
        report = ReviewReport(result=AuditResult(is_passed=True, score=90, summary="section ok"))
        return self._save_artifacts(
            {"section_review": report},
            step_name="validate_section",
            primary_key="section_review",
            context_key=None,
            summary="Validated section.",
        ).model_dump()

    def validate_output(self, output, context):
        self.calls.append(("validate_output", output, context))
        report = ReviewReport(result=AuditResult(is_passed=True, score=92, summary="output ok"))
        return self._save_artifacts(
            {"review_report": report},
            step_name="validate_output",
            primary_key="review_report",
            context_key=None,
            summary="Validated output.",
        ).model_dump()


def test_naive_writer_workflow_write_calls_main_tool_chain_and_writes_artifacts():
    calls = []
    d = os.path.join(os.path.dirname(__file__), "writer_workflow_artifacts")
    os.makedirs(d, exist_ok=True)
    workflow = NaiveWriterWorkflow(
        resource_tools=RecordingResourceTools(calls, artifact_store=d),
        context_tools=RecordingContextTools(calls, artifact_store=d),
        planning_tools=RecordingPlanningTools(calls, artifact_store=d),
        drafting_tools=RecordingDraftingTools(calls, artifact_store=d),
        quality_tools=RecordingQualityTools(calls, artifact_store=d),
    )

    result = workflow.write(
        task={"task_id": "task-1", "query": "写方案", "task_type": "write"},
        input_resources=[{"resource_id": "r1"}],
    )

    assert [call[0] for call in calls] == [
        "profile_resources",
        "create_writing_context",
        "generate_outline",
        "generate_section_instructions",
        "generate_draft_section",
        "validate_section",
        "update_writing_context",
        "generate_draft_document",
        "generate_writing_output",
        "validate_output",
    ]

    assert calls[1][2].endswith("resource_profiles.json")
    assert calls[2][2].endswith("writing_context.json")
    assert calls[2][3].endswith("resource_profiles.json")
    assert calls[4][2].endswith("section_instructions.json")
    assert calls[6][1].endswith("draft_section.json")
    assert calls[7][1].endswith("draft_section.json")
    assert calls[8][1].endswith("draft_document.json")

    expected_stage_keys = [
        "resource_profiles",
        "writing_context",
        "outline",
        "section_instructions",
        "draft_section",
        "section_review",
        "draft_document",
        "writing_output",
        "output_review",
    ]
    for stage_key in expected_stage_keys:
        stage_result = result["stage_results"][stage_key]
        assert os.path.exists(stage_result["artifact_path"])

    output = load_artifact_json(result["primary_result"]["artifact_path"], WritingOutput)
    assert output.content.startswith("# 方案")

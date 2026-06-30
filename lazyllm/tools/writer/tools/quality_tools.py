from __future__ import annotations
from typing import Any, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.quality import AuditResult, ReviewReport
from ..data_models.task import WritingOutput
from ..data_models.writing import SectionInstruction, SectionInstructionList
from ..utils import to_prompt_json

VALIDATE_SECTION_PROMPT = """你是一个章节质量审核员。请根据章节写作指令和写作上下文，对给定的章节草稿进行逐项检查，返回 AuditResult。

检查要求和规则：

1. 事实准确性（category=evidence）：将草稿中的具体数据、百分比、模型名称、事实性陈述逐一与指令中的 fact_constraints 和上下文中的 facts 进行比对。任何一处数值或事实错误 → severity=high，category=evidence。

2. 指令完整性（category=coverage）：检查草稿是否覆盖了指令中 required_points 的全部要点，是否完成了 section_goal。如果指令或上下文中指定了 must_include / must_avoid，逐一检查。缺失部分要点 → severity=medium，category=coverage。严重不完整（缺失一半以上要点）→ severity=high，category=coverage。

3. 结构对齐（category=format）：检查草稿的段落结构是否符合指令中的 expected_blocks 规划。标题层级错乱、遗漏承诺的子章节等重大结构偏差 → severity=medium，category=format。

4. 风格一致性（category=style）：检查语气、视角、正式程度是否与指令中的 style_constraints 和上下文中的 style_profile 一致。明显偏离 → severity=medium，category=style。

5. AI 味检测（category=style）：扫描全文，检测以下表达和句式模式：
   禁用词/表达（每出现一次 → severity=medium，category=style）：
   "在当今时代"、"在当今信息爆炸的时代"、"在当今……的时代"、"综上所述"、"不可否认的是"、"众所周知"、"值得一提的是"、"毋庸置疑"、"随着……的不断发展"、"具有重要的理论意义和现实意义"、"大势所趋"、"赋能"
   句式重复检测（→ severity=low，category=style）：
   - "首先……其次……最后……" 模式在连续 3 段及以上出现
   - "不仅……而且……" 作为段落开头
   - "此外" 作为段落开头出现超过 2 次

6. 引用完整性（category=evidence）：如果指令中指定了 source_refs，检查草稿是否适当引用了这些来源。缺失引用 → severity=low，category=evidence。

7. 篇幅检查（category=format）：如果指令中给出了字数限制（min_words / max_words），估算草稿的大致字数并标记偏差。→ severity=low，category=format。

评分规则：
- is_passed：存在任何 severity=high 的 issue，或者 severity=medium 的 issue 超过 3 个 → false。其余情况 → true。
- score：从 100 分起扣，每个 high -20，每个 medium -10，每个 low -3，最低 0。
- summary：用中文写一句话总体评估，点明主要发现。
- issues：逐条列出所有问题。无问题则为空列表。

草稿内容：
{section_json}

章节指令：
{instruction_json}

写作上下文：
{context_json}
"""


VALIDATE_OUTPUT_PROMPT = """你是一个终稿质量审核员。请对以下写作终稿进行逐项检查，返回 AuditResult。

终稿已经过章节级校验，本阶段重点检查：输出格式、跨章节事实一致性、AI 痕迹深度检测、风格一致性、引用完整性、读者适配性和结论质量。

检查要求和规则：

1. 输出格式（category=format）：检查 output_format 是否为声明的格式（通常为 markdown），content 中是否包含正确的标题层级（# 报告标题、## 章节标题、### 子章节标题），层级是否与上下文中的大纲结构一致。格式错误或层级错乱 → severity=high，category=format。references 字段缺失或不完整（少于 2 条）→ severity=high，category=format。

2. 跨章节事实一致性（category=evidence）：将全文中的数据点与上下文中的 facts 逐一比对。同一个数据在不同章节出现时数值必须一致，不得前后矛盾。出现矛盾 → severity=high，category=evidence。数据与 facts 中的锁定事实不一致 → severity=high，category=evidence。

3. AI 味深度检测（category=style）：
   a) 以下禁用表达每出现一次 → severity=medium，category=style：
      "在当今时代"、"在当今信息爆炸的时代"、"综上所述"、"不可否认的是"、"众所周知"、"值得一提的是"、"毋庸置疑"、"随着……的不断发展"、"具有重要的理论意义和现实意义"、"大势所趋"、"赋能"
   b) 空洞总结句检测（→ severity=low，category=style）：如"深度学习是一种强大的工具，将在未来发挥越来越重要的作用"、"人工智能是未来的发展方向"等无实质信息的泛泛之谈。
   c) 句式重复检测（→ severity=low，category=style）：
      - "首先……其次……最后……" 模式在连续 3 段及以上出现
      - "不仅……而且……" 作为段落开头
      - "此外" 作为段落开头出现超过 2 次

4. 风格一致性（category=style）：检查全文的语气、视角、正式程度是否与上下文中的 style_profile 一致。全文不同章节之间风格应统一，不应出现某章学术化、某章口语化的不一致。明显偏离或风格不一致 → severity=medium，category=style。

5. 引用完整性（category=evidence）：文中提到的每个模型、数据、观点是否标注了出处（作者, 年份）。参考文献是否至少包含上下文 facts 中引用的所有来源。缺少出处 → severity=medium，category=evidence。缺少关键来源 → severity=high，category=evidence。

6. 读者适配性（category=coverage）：根据上下文中的 style_profile.audience，检查专业术语是否在首次出现时给出解释（如面向非专业读者则必须有，面向专业读者可酌情省略）。术语未解释且面向非专业读者 → severity=medium，category=coverage。

7. 结论质量（category=style）：结论章节应以开放式问题或前瞻性展望结尾，而非确定性的总结论断。以"总之，……"或"综上所述，……"加确定性结论结尾 → severity=medium，category=style。

8. 来源归属（category=evidence）：全文不得以第一手研究姿态呈现（如"我们研究发现""本文原创性地提出"等），必须明确标注信息来源。缺失来源声明或姿态不当 → severity=medium，category=evidence。

评分规则：
- is_passed：存在任何 severity=high 的 issue，或者 severity=medium 的 issue 超过 3 个 → false。其余情况 → true。
- score：从 100 分起扣，每个 high -20，每个 medium -10，每个 low -3，最低 0。
- summary：用中文写一句话总体评估，点明主要发现。
- issues：逐条列出所有问题。无问题则为空列表。

终稿内容：
{output_json}

写作上下文：
{context_json}
"""


class WriterQualityTools(WriterToolBase):
    __public_apis__ = [
        "validate_section",
        "validate_output",
    ]

    def validate_section(
        self,
        draft_section: Any,
        section_instruction: Any,
        context: Any,
    ) -> dict:
        section_data = self._unified_raw_data(draft_section)
        instruction_list = self._unified_model(section_instruction, SectionInstructionList)
        writing_context = self._unified_model(context, WritingContext)

        instruction = self._match_instruction(section_data or {}, instruction_list)

        if instruction is None:
            fallback = AuditResult(
                is_passed=True,
                score=100,
                summary="未找到匹配的章节指令，跳过详细校验。",
                issues=[],
            )
            report = ReviewReport(
                target=section_data.get("section_id") or section_data.get("title") if section_data else None,
                result=fallback,
            )
            result = self._save_artifacts(
                {"section_review": report},
                step_name="validate_section",
                primary_key="section_review",
                summary="Section validation skipped: no matching instruction.",
                counts={"total_issues": 0, "high_severity": 0, "medium_severity": 0, "low_severity": 0},
                artifact_meta={"is_passed": True, "score": 100, "match_found": False},
            )
            return result.model_dump()

        prompt = VALIDATE_SECTION_PROMPT.format(
            section_json=to_prompt_json(section_data),
            instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
        )

        audit_result = self._call_llm_structured(prompt, AuditResult)

        section_title = section_data.get("title") if section_data else None
        section_id = section_data.get("section_id") if section_data else None

        report = ReviewReport(
            target=section_id or section_title or instruction.section_title or "unknown",
            result=audit_result,
            meta={
                "instruction_id": instruction.instruction_id,
                "outline_node_id": instruction.outline_node_id,
                "section_title": instruction.section_title,
            },
        )

        high_count = sum(1 for i in audit_result.issues if i.severity == "high")
        medium_count = sum(1 for i in audit_result.issues if i.severity == "medium")
        low_count = sum(1 for i in audit_result.issues if i.severity == "low")

        result = self._save_artifacts(
            {"section_review": report},
            step_name="validate_section",
            primary_key="section_review",
            summary=f"Section validation: {'PASSED' if audit_result.is_passed else 'FAILED'} (score: {audit_result.score}/100)",
            counts={
                "total_issues": len(audit_result.issues),
                "high_severity": high_count,
                "medium_severity": medium_count,
                "low_severity": low_count,
            },
            artifact_meta={
                "section_id": section_id,
                "section_title": section_title,
                "instruction_id": instruction.instruction_id,
                "is_passed": audit_result.is_passed,
                "score": audit_result.score,
            },
        )
        return result.model_dump()

    def validate_output(
        self,
        output: Any,
        context: Any,
    ) -> dict:
        writing_output = self._unified_model(output, WritingOutput)
        writing_context = self._unified_model(context, WritingContext)

        prompt = VALIDATE_OUTPUT_PROMPT.format(
            output_json=to_prompt_json(writing_output),
            context_json=to_prompt_json(writing_context),
        )

        audit_result = self._call_llm_structured(prompt, AuditResult)

        report = ReviewReport(
            target=writing_output.output_id or writing_output.title or "untitled",
            result=audit_result,
            meta={
                "output_id": writing_output.output_id,
                "output_title": writing_output.title,
                "output_format": writing_output.output_format,
                "context_id": writing_context.context_id,
            },
        )

        high_count = sum(1 for i in audit_result.issues if i.severity == "high")
        medium_count = sum(1 for i in audit_result.issues if i.severity == "medium")
        low_count = sum(1 for i in audit_result.issues if i.severity == "low")

        result = self._save_artifacts(
            {"output_review": report},
            step_name="validate_output",
            primary_key="output_review",
            summary=f"Output validation: {'PASSED' if audit_result.is_passed else 'FAILED'} (score: {audit_result.score}/100)",
            counts={
                "total_issues": len(audit_result.issues),
                "high_severity": high_count,
                "medium_severity": medium_count,
                "low_severity": low_count,
            },
            artifact_meta={
                "output_id": writing_output.output_id,
                "output_title": writing_output.title,
                "is_passed": audit_result.is_passed,
                "score": audit_result.score,
            },
        )
        return result.model_dump()

    def _match_instruction(
        self,
        section_data: dict,
        instruction_list: SectionInstructionList,
    ) -> Optional[SectionInstruction]:
        section_instruction_id = section_data.get("instruction_id") or ""
        section_node_id = section_data.get("outline_node_id") or ""
        section_title = section_data.get("title") or ""

        for inst in instruction_list.instructions:
            if section_instruction_id and inst.instruction_id == section_instruction_id:
                return inst
            if section_node_id and inst.outline_node_id == section_node_id:
                return inst

        for inst in instruction_list.instructions:
            if section_title and inst.section_title == section_title:
                return inst

        for section_block in section_data.get("blocks") or []:
            block_heading = (section_block.get("heading") or "").strip()
            if block_heading:
                for inst in instruction_list.instructions:
                    if block_heading == inst.section_title:
                        return inst

        return None

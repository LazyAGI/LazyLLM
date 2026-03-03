from lazyllm import pipeline
from lazyllm.tools.data import codegen_ops


def build_codegen_pipeline(model, input_key='messages', min_score=7, max_score=10):
    with pipeline() as ppl:
        ppl.code_instruction_generator = codegen_ops.CodeInstructionGenerator(
            model=model,
            input_key=input_key,
            output_key='instruction'
        )

        ppl.script_synthesizer = codegen_ops.ScriptSynthesizer(
            model=model,
            input_key='instruction',
            output_key='new_code'
        )

        ppl.logic_integrity_auditor = codegen_ops.LogicIntegrityAuditor(
            model=model,
            input_instruction_key='instruction',
            input_code_key='new_code',
            output_score_key='quality_score',
            output_feedback_key='feedback'
        )

        ppl.threshold_sieve = codegen_ops.ThresholdSieve(
            min_score=min_score,
            max_score=max_score,
            input_score_key='quality_score',
            output_key='quality_score_filter_label'
        )

        ppl.code_feedback_formatter = codegen_ops.CodeFeedbackFormatter(
            instruction_key='messages',
            input_code_key='new_code',
            feedback_key='feedback',
            output_key='formatted_data'
        )

    return ppl


def build_simple_codegen_pipeline(model, input_key='messages'):
    with pipeline() as ppl:
        ppl.code_instruction_generator = codegen_ops.CodeInstructionGenerator(
            model=model,
            input_key=input_key,
            output_key='instruction'
        )

        ppl.script_synthesizer = codegen_ops.ScriptSynthesizer(
            model=model,
            input_key='instruction',
            output_key='new_code'
        )

    return ppl

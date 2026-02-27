from lazyllm import pipeline
from lazyllm.tools.data import codegen_ops


def build_codegen_pipeline(model, input_key='messages', min_score=7, max_score=10):
    with pipeline() as ppl:
        ppl.code_instruction_generator = codegen_ops.CodeInstructionGenerator(
            model=model,
            input_key=input_key
        )
        ppl.script_synthesizer = codegen_ops.ScriptSynthesizer(model=model)
        ppl.threshold_sieve = codegen_ops.ThresholdSieve(
            model=model,
            min_score=min_score,
            max_score=max_score
        )
    return ppl

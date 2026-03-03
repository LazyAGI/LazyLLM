from lazyllm import pipeline
from lazyllm.tools.data import text2sql_ops


def build_text2sql_full_pipeline(model, database_manager, embedding_model=None,
                                 output_num=300, num_generations=10, input_query_num=5):
    with pipeline() as ppl:
        ppl.sql_forge = text2sql_ops.SQLForge(
            model=model,
            database_manager=database_manager,
            output_num=output_num
        )
        ppl.sql_runtime_sieve = text2sql_ops.SQLRuntimeSieve(
            database_manager=database_manager
        )
        ppl.sql_intent_synthesizer = text2sql_ops.SQLIntentSynthesizer(
            model=model,
            embedding_model=embedding_model,
            database_manager=database_manager,
            input_query_num=input_query_num
        )
        ppl.tsql_semantic_auditor = text2sql_ops.TSQLSemanticAuditor(
            model=model,
            database_manager=database_manager
        )
        ppl.sql_context_assembler = text2sql_ops.SQLContextAssembler(
            database_manager=database_manager
        )
        ppl.sql_reasoning_tracer = text2sql_ops.SQLReasoningTracer(
            model=model,
            database_manager=database_manager
        )
        ppl.sql_consensus_unifier = text2sql_ops.SQLConsensusUnifier(
            database_manager=database_manager
        )
        ppl.sql_syntax_profiler = text2sql_ops.SQLSyntaxProfiler()
        ppl.sql_effort_ranker = text2sql_ops.SQLEffortRanker(
            model=model,
            database_manager=database_manager,
            num_generations=num_generations
        )
    return ppl

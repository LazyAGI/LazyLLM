from lazyllm import pipeline
from lazyllm.tools.data import text2sql_ops


def text2sql_synthetic_ppl(model, database_manager, embedding_model=None,
                           output_num=300, num_generations=10, input_query_num=5,
                           output_format='alpaca', target_complexity='hard'):
    with pipeline() as ppl:
        ppl.sql_forge = text2sql_ops.SQLForge(
            model=model,
            database_manager=database_manager,
            output_num=output_num,
            target_complexity=target_complexity
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
        if output_format:
            ppl.formatter = text2sql_ops.Text2SQLToSFTFormatter(
                format_type=output_format
            )
    return ppl


def text2sql_enhanced_ppl(model, database_manager, embedding_model=None,
                          output_num=3, num_generations=10,
                          output_format='alpaca', target_complexity='hard'):
    with pipeline() as ppl:
        ppl.sql_question_generator = text2sql_ops.SQLQuestionGenerator(
            model=model,
            database_manager=database_manager,
            output_num=output_num,
            target_complexity=target_complexity
        )
        ppl.sql_context_assembler = text2sql_ops.SQLContextAssembler(
            database_manager=database_manager
        )
        ppl.sql_generator = text2sql_ops.SQLGenerator(
            model=model,
            database_manager=database_manager,
            output_num=1,
            target_complexity=target_complexity
        )
        ppl.sql_runtime_sieve = text2sql_ops.SQLRuntimeSieve(
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
        if output_format:
            ppl.formatter = text2sql_ops.Text2SQLToSFTFormatter(
                format_type=output_format
            )
    return ppl

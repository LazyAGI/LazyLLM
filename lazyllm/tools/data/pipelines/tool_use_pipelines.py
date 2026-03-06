from lazyllm import pipeline
from lazyllm.tools.data import tool_use_ops


def build_tool_use_pipeline(model, input_key='content', n_turns=6):
    with pipeline() as ppl:
        ppl.contextual_beacon = tool_use_ops.ContextualBeacon(
            model=model,
            input_key=input_key,
            output_key='scenario'
        )
        ppl.scenario_diverger = tool_use_ops.ScenarioDiverger(
            model=model,
            input_key='scenario',
            output_key='expanded_scenarios'
        )
        ppl.decomposition_kernel = tool_use_ops.DecompositionKernel(
            model=model,
            input_key='scenario',
            output_key='atomic_tasks'
        )
        ppl.chained_logic_assembler = tool_use_ops.ChainedLogicAssembler(
            model=model,
            input_key='atomic_tasks',
            output_key='sequential_tasks'
        )
        ppl.topology_architect = tool_use_ops.TopologyArchitect(
            model=model,
            input_key='atomic_tasks',
            output_key='para_seq_tasks'
        )
        ppl.viability_sieve = tool_use_ops.ViabilitySieve(
            model=model,
            input_composition_key='para_seq_tasks',
            input_atomic_key='atomic_tasks',
            output_key='filtered_composition_tasks'
        )
        ppl.protocol_specifier = tool_use_ops.ProtocolSpecifier(
            model=model,
            input_composition_key='filtered_composition_tasks',
            input_atomic_key='atomic_tasks',
            output_key='functions'
        )
        ppl.dialogue_simulator = tool_use_ops.DialogueSimulator(
            model=model,
            input_composition_key='filtered_composition_tasks',
            input_functions_key='functions',
            output_key='conversation',
            n_turns=n_turns
        )
    return ppl


def build_simple_tool_use_pipeline(model, input_key='content', n_tasks=5, n_turns=6):
    with pipeline() as ppl:
        ppl.contextual_beacon = tool_use_ops.ContextualBeacon(
            model=model,
            input_key=input_key,
            output_key='scenario'
        )
        ppl.decomposition_kernel = tool_use_ops.DecompositionKernel(
            model=model,
            input_key='scenario',
            output_key='atomic_tasks',
            n=n_tasks
        )
        ppl.protocol_specifier = tool_use_ops.ProtocolSpecifier(
            model=model,
            input_composition_key='atomic_tasks',
            input_atomic_key='atomic_tasks',
            output_key='functions'
        )
        ppl.dialogue_simulator = tool_use_ops.DialogueSimulator(
            model=model,
            input_composition_key='atomic_tasks',
            input_functions_key='functions',
            output_key='conversation',
            n_turns=n_turns
        )
    return ppl

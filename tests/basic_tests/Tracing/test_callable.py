from lazyllm import enable_trace


def test_enable_trace_function_call(exporter):
    def add(a, b): return a + b

    result = enable_trace(add, 5, 3)
    assert result == 8

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == 'add'
    assert spans[0].attributes.get('lazyllm.span.kind') == 'callable'
    assert spans[0].attributes.get('lazyllm.status') == 'ok'
    assert spans[0].attributes.get('lazyllm.io.output') == '8'
    assert spans[0].attributes.get('lazyllm.entity.name') == 'add'


def test_enable_trace_decorator(exporter):
    @enable_trace()
    def subtract(x, y): return x - y

    result = subtract(10, 3)

    spans = exporter.get_finished_spans()
    assert result == 7
    assert len(spans) == 1 and spans[0].name == 'subtract'
    assert spans[0].attributes.get('lazyllm.span.kind') == 'callable'
    assert spans[0].attributes.get('lazyllm.entity.name') == 'subtract'


def test_enable_trace_lambda(exporter):
    result = enable_trace(lambda x: x * 2, 5)
    assert result == 10

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == '<lambda>'
    assert spans[0].attributes.get('lazyllm.span.kind') == 'callable'
    assert spans[0].attributes.get('lazyllm.entity.name') == '<lambda>'


def test_enable_trace_agent_uses_gen_ai_agent_semantics_and_omits_payload(exporter):
    class AgentInvocation:
        __span_name__ = 'invoke_agent'
        _type = 'agent'
        _agent_name = 'ChatAgent'

        def __init__(self):
            self._agent_name = 'ChatAgent'

        def __call__(self, prompt):
            return f'answer:{prompt}'

    result = enable_trace(
        AgentInvocation(),
        'private user prompt',
        trace_id='agent-trace',
        session_id='conversation-1',
        debug_capture_payload=False,
    )

    spans = exporter.get_finished_spans()
    assert result == 'answer:private user prompt'
    assert len(spans) == 1 and spans[0].name == 'invoke_agent'
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'agent'
    assert spans[0].attributes.get('gen_ai.operation.name') == 'invoke_agent'
    assert spans[0].attributes.get('gen_ai.agent.name') == 'ChatAgent'
    assert spans[0].attributes.get('gen_ai.conversation.id') == 'conversation-1'
    assert 'lazyllm.io.input' not in spans[0].attributes
    assert 'lazyllm.io.output' not in spans[0].attributes

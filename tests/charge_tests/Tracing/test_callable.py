from lazyllm import enable_trace


def test_simple_function_tracing(exporter):
    def add(a, b):
        return a + b

    result = enable_trace(add, 5, 3)
    assert result == 8

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == "add"
    assert spans[0].attributes.get("lazyllm.span.kind") == "callable"
    assert spans[0].attributes.get("lazyllm.status") == "ok"
    assert spans[0].attributes.get("lazyllm.io.output") == "8"
    assert spans[0].attributes.get("lazyllm.entity.name") == "add"


def test_decorator_tracing(exporter):
    @enable_trace()
    def subtract(x, y):
        return x - y

    result = subtract(10, 3)

    spans = exporter.get_finished_spans()
    assert result == 7
    assert len(spans) == 1 and spans[0].name == "subtract"
    assert spans[0].attributes.get("lazyllm.span.kind") == "callable"
    assert spans[0].attributes.get("lazyllm.entity.name") == "subtract"


def test_lambda_function_tracing(exporter):
    func = lambda x: x * 2
    result = enable_trace(func, 5)
    assert result == 10

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == "<lambda>"
    assert spans[0].attributes.get("lazyllm.span.kind") == "callable"
    assert spans[0].attributes.get("lazyllm.entity.name") == "<lambda>"


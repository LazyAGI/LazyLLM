from lazyllm.tools.data import (build_pre_suffix, process_uppercase,
                                AddSuffix, rich_content)


class TestDataOperators:

    def test_build_pre_suffix(self):
        func = build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
        inputs = [{'text': 'world'}, {'text': 'lazyLLM'}]
        res = func(inputs)
        assert res == [{'text': 'Hello, world!'}, {'text': 'Hello, lazyLLM!'}]

    def test_process_uppercase(self):
        func = process_uppercase(input_key='text')
        inputs = [{'text': 'hello'}, {'text': 'world'}]
        res = func(inputs)
        assert res == [{'text': 'HELLO'}, {'text': 'WORLD'}]

    def test_add_suffix(self):
        func = AddSuffix(input_key='text', suffix='!!!', _max_workers=2)
        inputs = [{'text': 'exciting'}, {'text': 'amazing'}]
        res = func(inputs)
        assert res == [{'text': 'exciting!!!'}, {'text': 'amazing!!!'}]

    def test_rich_content(self):
        func = rich_content(input_key='text')
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        print(res)
        assert res == [
            {'text': 'This is a test.'},
            {'text': 'This is a test. - part 1'},
            {'text': 'This is a test. - part 2'}]

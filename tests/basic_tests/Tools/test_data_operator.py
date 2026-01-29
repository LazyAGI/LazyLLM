from lazyllm.tools.data import demo1, demo2

class TestDataOperators:

    def test_build_pre_suffix(self):
        func = demo1.build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
        inputs = [{'text': 'world'}, {'text': 'lazyLLM'}]
        res = func(inputs)
        assert res == [{'text': 'Hello, world!'}, {'text': 'Hello, lazyLLM!'}]

    def test_process_uppercase(self):
        func = demo1.process_uppercase(input_key='text')
        inputs = [{'text': 'hello'}, {'text': 'world'}]
        res = func(inputs)
        assert res == [{'text': 'HELLO'}, {'text': 'WORLD'}]

    def test_add_suffix(self):
        func = demo2.AddSuffix(input_key='text', suffix='!!!', _max_workers=2)
        inputs = [{'text': 'exciting'}, {'text': 'amazing'}]
        res = func(inputs)
        assert res == [{'text': 'exciting!!!'}, {'text': 'amazing!!!'}]

    def test_rich_content(self):
        func = demo2.rich_content(input_key='text')
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        print(res)
        assert res == [
            {'text': 'This is a test.'},
            {'text': 'This is a test. - part 1'},
            {'text': 'This is a test. - part 2'}]

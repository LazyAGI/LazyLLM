from lazyllm.tools.data.pipelines.demo_pipelines import build_demo_pipeline

class TestDataPipeline:
    def test_demo_pipeline(self):
        ppl = build_demo_pipeline()
        data = [{'text': 'lazyLLM'} for _ in range(2)]
        res = ppl(data)
        print(res)
        assert len(res) == 6
        assert res == [
            {'text': 'HELLO, LAZYLLM!!!!'},
            {'text': 'HELLO, LAZYLLM!!!! - part 1'},
            {'text': 'HELLO, LAZYLLM!!!! - part 2'},
            {'text': 'HELLO, LAZYLLM!!!!'},
            {'text': 'HELLO, LAZYLLM!!!! - part 1'},
            {'text': 'HELLO, LAZYLLM!!!! - part 2'}]

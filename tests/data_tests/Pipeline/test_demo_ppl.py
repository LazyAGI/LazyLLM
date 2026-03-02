import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.demo_pipelines import build_demo_pipeline

class TestDataPipeline:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_demo_pipeline(self):
        ppl = build_demo_pipeline()
        data = [{'text': 'lazyLLM'} for _ in range(2)]
        res = ppl(data)
        assert len(res) == 6
        assert res == [
            {'text': 'HELLO, LAZYLLM!!!!'},
            {'text': 'HELLO, LAZYLLM!!!! - part 1'},
            {'text': 'HELLO, LAZYLLM!!!! - part 2'},
            {'text': 'HELLO, LAZYLLM!!!!'},
            {'text': 'HELLO, LAZYLLM!!!! - part 1'},
            {'text': 'HELLO, LAZYLLM!!!! - part 2'}]

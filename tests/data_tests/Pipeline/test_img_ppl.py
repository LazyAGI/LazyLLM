import os
import shutil
import tempfile
from lazyllm import config
from lazyllm.tools.data.pipelines.img_pipelines import build_img2qa_pipeline

class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self


class TestImgQAPipeline:

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

    def test_img2qa_pipeline(self):

        excepted_output = {
            'query': 'what is the animal in the pic?',
            'answer': 'Reasoning: Upon vxxxxxxxx',
            'score': 1
        }

        model = MockModel(excepted_output)

        ppl = build_img2qa_pipeline(
            model=model,
            filter_threshold=1,
            image_key='image',
            context_key='context',
            img_resize=True,
            to_chat=True
        )

        data_path = config['data_path']
        test_img = os.path.join(data_path, 'ci_data/dog.png')
        data = {'image': test_img,
                'context': 'what is the animal in the pic?',
                'reference': 'a dog'
        }

        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 1
        assert 'messages' in res[0]
        assert 'images' in res[0]

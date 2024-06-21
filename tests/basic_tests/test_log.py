import os
import lazyllm
import pytest

@pytest.fixture(scope='class')
def set_env_vars(request):
    os.environ['LAZYLLM_DEBUG'] = '1'
    os.environ['LAZYLLM_LOG_LEVEL'] = 'INFO'
    os.environ['LAZYLLM_LOG_DIR'] = '~/.lazyllm'
    os.environ['LAZYLLM_LOG_FILE_SIZE'] = '1 MB'

    def teardown():
        del os.environ['LAZYLLM_DEBUG']
        del os.environ['LAZYLLM_LOG_LEVEL']
        del os.environ['LAZYLLM_LOG_DIR']
        del os.environ['LAZYLLM_LOG_FILE_SIZE']

    request.addfinalizer(teardown)

class TestLazyLLM(object):

    @pytest.mark.usefixtures('set_env_vars')
    def test_debug_env_var(self):
        assert os.environ.get('LAZYLLM_DEBUG') == '1'
        assert os.environ.get('LAZYLLM_LOG_LEVEL') == 'INFO'
        assert os.environ.get('LAZYLLM_LOG_DIR') == '~/.lazyllm'
        assert os.environ.get('LAZYLLM_LOG_FILE_SIZE') == '1 MB'


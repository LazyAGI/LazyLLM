import os

from lazyllm import finetune, launchers

class TestFinetune(object):

    def test_finetune_alpacalora(self):
        # test instantiation
        f = finetune.alpacalora(base_model='internlm2-chat-7b', target_path='')
        assert f.base_model == 'internlm2-chat-7b'

    def test_finetune_collie(self):
        # test instantiation
        f = finetune.collie(base_model='internlm2-chat-7b', target_path='')
        assert f.base_model == 'internlm2-chat-7b'

    def test_auto_finetune(self):
        # test instantiation
        m = finetune.auto('internlm2-chat-7b', '', launcher=launchers.sco(ngpus=1))
        assert isinstance(m._launcher, launchers.sco)
        assert os.path.exists(m.base_model)

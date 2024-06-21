import os
import json

import lazyllm
from lazyllm import finetune, deploy, launchers


class TestFn_Component(object):
    def test_prompter(self):
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        assert not p.is_empty(), "Prompter should not be empty"

    def test_generate_prompt(self):
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        result = p.generate_prompt('123')
        assert result == 'hello world2 <123>', f"Expected 'hello world2 <123>', but got '{result}'"

    def test_generate_prompt_dict_input(self):
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        result_dict_input = p.generate_prompt({'input': '123'})
        assert result_dict_input == 'hello world2 <123>', f"Expected 'hello world2 <123>', but got '{result_dict_input}'"

    def test_from_template(self):
        p = lazyllm.Prompter.from_template('alpaca')
        expected_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        assert p._prompt == expected_prompt, f"Expected prompt to be '{expected_prompt}', but got '{p._prompt}'"

    def test_generate_prompt_with_template(self):
        p = lazyllm.Prompter.from_template('alpaca')
        result = p.generate_prompt(dict(instruction='ins', input='inp'))
        expected_result = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\nins\n\n### Input:\ninp\n\n### Response:\n"
        )
        assert result == expected_result, f"Expected '{expected_result}', but got '{result}'"

    def test_finetune_alpacalora(self):
        # test instantiation
        f = finetune.alpacalora(base_model='internlm2-chat-7b', target_path='')
        assert f.base_model == 'internlm2-chat-7b'

    def test_finetune_collie(self):
        # test instantiation
        f = finetune.collie(base_model='internlm2-chat-7b', target_path='')
        assert f.base_model == 'internlm2-chat-7b'

    def test_deploy_lightllm(self):
        # test instantiation
        m = deploy.lightllm(trust_remote_code=False, launcher=launchers.sco)
        assert m.trust_remote_code == False
        assert type(m.launcher) == launchers.sco

    def test_deploy_vllm(self):
        # test instantiation
        m = deploy.vllm(trust_remote_code=False, launcher=launchers.sco)
        assert m.trust_remote_code == False
        assert type(m.launcher) == launchers.sco

    def test_auto_finetune(self):
        # test instantiation
        m = finetune.auto('internlm2-chat-7b', '', launcher=launchers.sco(ngpus=1))
        assert type(m.launcher) == launchers.sco
        assert os.path.exists(m.base_model)

    def test_auto_deploy(self):
        # test instantiation
        m = deploy.auto('internlm2-chat-7b', trust_remote_code=False, launcher=launchers.sco(ngpus=1))
        assert m.trust_remote_code == False
        assert type(m.launcher) == launchers.sco

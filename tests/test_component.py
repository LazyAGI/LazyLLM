import lazyllm
from lazyllm import finetune, deploy


class TestFn_Component(object):
    def test_prompter(self):
        # 创建Prompter对象
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        # 检查Prompter对象是否为空
        assert not p.is_empty(), "Prompter should not be empty"

    def test_generate_prompt(self):
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        # 测试generate_prompt方法
        result = p.generate_prompt('123')
        assert result == 'hello world2 <123>', f"Expected 'hello world2 <123>', but got '{result}'"

    def test_generate_prompt_dict_input(self):
        p = lazyllm.Prompter(prompt='hello world2 <{input}>')
        # 测试generate_prompt方法，使用字典输入
        result_dict_input = p.generate_prompt({'input': '123'})
        assert result_dict_input == 'hello world2 <123>', f"Expected 'hello world2 <123>', but got '{result_dict_input}'"

    def test_from_template(self):
        # 使用from_template方法创建Prompter对象
        p = lazyllm.Prompter.from_template('alpaca')
        expected_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        assert p._prompt == expected_prompt, f"Expected prompt to be '{expected_prompt}', but got '{p._prompt}'"

    def test_generate_prompt_with_template(self):
        p = lazyllm.Prompter.from_template('alpaca')
        # 测试generate_prompt方法，使用模板和字典输入
        result = p.generate_prompt(dict(instruction='ins', input='inp'))
        expected_result = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\nins\n\n### Input:\ninp\n\n### Response:\n"
        )
        assert result == expected_result, f"Expected '{expected_result}', but got '{result}'"

    def test_finetune_alpacalora(self):
        f = finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1')
        # 测试finetune.alpacalora方法
        assert f.target_path == './finetune-target1/lora', f"Expected target_path to be './finetune-target1', but got '{f.target_path}'"

    def test_finetune_collie(self):
        f = finetune.collie(base_model='./base-model2', target_path='./finetune-target2')
        # 测试finetune.collie方法
        assert f.target_path == './finetune-target2', f"Expected target_path to be './finetune-target2', but got '{f.target_path}'"

    def test_deploy_lightllm(self):
        d = deploy.lightllm(
            launcher=lazyllm.launchers.slurm(
                partition='pat_rd',
                nnode=1,
                nproc=1,
                ngpus=1,
                sync=False
            ),
        )
        assert d.geturl() == 'http://{ip}:{port}/generate', f"Expected 'http://ip:port/generate', but got '{d.geturl()}'"

    def test_deploy_vllm(self):
        d = deploy.vllm(
            launcher=lazyllm.launchers.slurm(
                partition='pat_rd',
                nnode=1,
                nproc=1,
                ngpus=1,
                sync=False
            ),
        )
        assert d.geturl() == 'http://{ip}:{port}/generate', f"Expected 'http://ip:port/generate', but got '{d.geturl()}'"

    def test_auto_finetune(self):
        # Not implemented
        raise NotImplementedError

    def test_auto_deploy(self):
        # Not implemented
        raise NotImplementedError

    def test_embedding(self):
        # Not implemented
        raise NotImplementedError

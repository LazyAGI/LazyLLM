import lazyllm


class TestPrompter(object):
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
        assert result_dict_input == 'hello world2 <123>', \
               f"Expected 'hello world2 <123>', but got '{result_dict_input}'"

    def test_from_template(self):
        p = lazyllm.Prompter.from_template('alpaca')
        expected_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### "
            "Input:\n{input}\n\n### Response:\n"
        )
        assert p._prompt == expected_prompt, f"Expected prompt to be '{expected_prompt}', but got '{p._prompt}'"

    def test_generate_prompt_with_template(self):
        p = lazyllm.Prompter.from_template('alpaca')
        result = p.generate_prompt(dict(instruction='ins', input='inp'))
        expected_result = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n### Instruction:\nins\n\n### "
            "Input:\ninp\n\n### Response:\n"
        )
        assert result == expected_result, f"Expected '{expected_result}', but got '{result}'"


class TestAlpacaPrompter(object):
    def test_basic_prompter(self):
        p = lazyllm.AlpacaPrompter('请完成加法运算, 输入为{instruction}')
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算, 输入为a+b\n\n\n\n### Response:\n'  # noqa E501
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算, 输入为a+b\n\n'},  # noqa E501
            {'role': 'user', 'content': ''}]}

        p = lazyllm.AlpacaPrompter('请完成加法运算', extra_keys='input')
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算\n\nHere are some extra messages you can referred to:\n\n### input:\na+b\n\n\n\n### Response:\n'  # noqa E501
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算\n\nHere are some extra messages you can referred to:\n\n### input:\na+b\n\n'},  # noqa E501
            {'role': 'user', 'content': ''}]}

        p = lazyllm.AlpacaPrompter(dict(system='请完成加法运算', user='输入为{instruction}'))
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算\n\n输入为a+b### Response:\n'  # noqa E501
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n### Instruction:\n请完成加法运算'}, {'role': 'user', 'content': '输入为a+b'}]}  # noqa E501


class TestChatPrompter(object):
    def test_basic_prompter(self):
        p = lazyllm.ChatPrompter('请完成加法运算, 输入为{instruction}')
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算, 输入为a+b\n\n\n\n\n\n\n\n'
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算, 输入为a+b\n\n'},
            {'role': 'user', 'content': ''}]}

        p = lazyllm.ChatPrompter('请完成加法运算', extra_keys='input')
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算\nHere are some extra messages you can referred to:\n\n### input:\na+b\n\n\n\n\n\n\n\n\n'  # noqa E501
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算\nHere are some extra messages you can referred to:\n\n### input:\na+b\n\n\n'},  # noqa E501
            {'role': 'user', 'content': ''}]}

        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'))
        r = p.generate_prompt('a+b')
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n\n\n输入为a+b\n\n'
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'},
            {'role': 'user', 'content': '输入为a+b'}]}

    def test_history(self):
        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'),
                                 history=[['输入为a+b', 'a+b'], ['输入为c+d', 'c+d']])
        r = p.generate_prompt('e+f')
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n输入为a+ba+b输入为c+dc+d\n\n输入为e+f\n\n'
        r = p.generate_prompt('e+f', return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'},
            {'role': 'user', 'content': '输入为a+b'},
            {'role': 'assistant', 'content': 'a+b'},
            {'role': 'user', 'content': '输入为c+d'},
            {'role': 'assistant', 'content': 'c+d'},
            {'role': 'user', 'content': '输入为e+f'}]}

        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'))
        r = p.generate_prompt('e+f', history=[['输入为a+b', 'a+b'], ['输入为c+d', 'c+d']])
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n输入为a+ba+b输入为c+dc+d\n\n输入为e+f\n\n'

        r = p.generate_prompt('e+f', history=[['输入为a+b', 'a+b'], ['输入为c+d', 'c+d']], return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'},
            {'role': 'user', 'content': '输入为a+b'},
            {'role': 'assistant', 'content': 'a+b'},
            {'role': 'user', 'content': '输入为c+d'},
            {'role': 'assistant', 'content': 'c+d'},
            {'role': 'user', 'content': '输入为e+f'}]}

        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'), history=[['输入为a+b', 'a+b']])
        r = p.generate_prompt('e+f', history=[{"role": "user", "content": '输入为c+d'},
                                              {"role": "assistant", "content": 'c+d'}])
        assert r == 'You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n输入为a+ba+b输入为c+dc+d\n\n输入为e+f\n\n'

        r = p.generate_prompt('e+f', history=[{"role": "user", "content": '输入为c+d'},
                                              {"role": "assistant", "content": 'c+d'}], return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'},
            {'role': 'user', 'content': '输入为a+b'},
            {'role': 'assistant', 'content': 'a+b'},
            {'role': 'user', 'content': '输入为c+d'},
            {'role': 'assistant', 'content': 'c+d'},
            {'role': 'user', 'content': '输入为e+f'}]}

    def test_empty_prompt_with_history(self):
        p = lazyllm.ChatPrompter('', history=[['输入为a+b', 'a+b']])
        r11 = p.generate_prompt('c+d')
        r12 = p.generate_prompt('c+d', return_dict=True)

        p = lazyllm.ChatPrompter(None, history=[['输入为a+b', 'a+b']])
        r21 = p.generate_prompt('c+d')
        r22 = p.generate_prompt('c+d', return_dict=True)

        assert r11 == r21 == 'You are an AI-Agent developed by LazyLLM.\n\n输入为a+ba+b\n\nc+d\n\n'
        assert r12 == r22 == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.'},
            {'role': 'user', 'content': '输入为a+b'},
            {'role': 'assistant', 'content': 'a+b'},
            {'role': 'user', 'content': 'c+d'}]}

    def test_configs(self):
        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'))
        p._set_model_configs(sos='<s>', eos='</s>')
        r = p.generate_prompt('a+b')
        assert r == '<s>You are an AI-Agent developed by LazyLLM.请完成加法运算</s>\n\n\n\n输入为a+b\n\n'
        r = p.generate_prompt('a+b', return_dict=True)
        assert r == {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'}, {'role': 'user', 'content': '输入为a+b'}]}  # noqa E501

    def test_config_with_history(self):
        p = lazyllm.ChatPrompter(dict(system='请完成加法运算', user='输入为{instruction}'), history=[['输入为a+b', 'a+b']])
        p._set_model_configs(sos='<s>', eos='</s>', soh='<h>', eoh='</h>', soa='<a>', eoa='</a>')
        r = p.generate_prompt('e+f', history=[{"role": "user", "content": '输入为c+d'},
                                              {"role": "assistant", "content": 'c+d'}])
        assert r == '<s>You are an AI-Agent developed by LazyLLM.请完成加法运算</s>\n\n<h>输入为a+b</h><a>a+b</a><h>输入为c+d</h><a>c+d</a>\n<h>\n输入为e+f\n</h><a>\n'  # noqa E501

        r = p.generate_prompt('e+f', history=[{"role": "user", "content": '输入为c+d'},
                                              {"role": "assistant", "content": 'c+d'}], return_dict=True)
        assert r == {'messages': [
            {'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\n请完成加法运算'},
            {'role': 'user', 'content': '输入为a+b'},
            {'role': 'assistant', 'content': 'a+b'},
            {'role': 'user', 'content': '输入为c+d'},
            {'role': 'assistant', 'content': 'c+d'},
            {'role': 'user', 'content': '输入为e+f'}]}

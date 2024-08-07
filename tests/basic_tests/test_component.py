import lazyllm


class TestComponent(object):
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

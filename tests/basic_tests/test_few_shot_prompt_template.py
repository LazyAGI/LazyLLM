import pytest
from lazyllm.prompt_templates import PromptTemplate, FewShotPromptTemplate
import textwrap


class TestFewShotPromptTemplate:

    def test_basic_creation(self):
        '''Test basic FewShotPromptTemplate creation'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a helpful assistant. Here are some examples:',
            suffix='Now please answer: {question}',
            examples=[
                {'input': 'What is 2+2?', 'output': '4'},
                {'input': 'What is 3+3?', 'output': '6'}
            ],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )
        assert few_shot_template.required_vars == ['question']

    def test_format_basic(self):
        '''Test basic formatting'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='Examples:',
            suffix='Question: {question}',
            examples=[
                {'input': 'What is 2+2?', 'output': '4'},
                {'input': 'What is 3+3?', 'output': '6'}
            ],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )

        result = few_shot_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            Examples:
            Input: What is 2+2?
            Output: 4
            Input: What is 3+3?
            Output: 6
            Question: What is 4+4?
            ''').strip()
        assert result == expected

    def test_format_with_prefix_and_suffix_variables(self):
        '''Test formatting with variables in both prefix and suffix'''
        egs_template = PromptTemplate.from_template('Q: {question}\nA: {answer}')
        few_shot_template = FewShotPromptTemplate(
            prefix='{role} Here are examples:',
            suffix='Now {action}: {question}',
            examples=[
                {'question': 'What is 2+2?', 'answer': '4'},
                {'question': 'What is 3+3?', 'answer': '6'}
            ],
            egs_prompt_template=egs_template,
            required_vars=['role', 'action', 'question'],
            partial_vars={}
        )

        result = few_shot_template.format(role='You are a calculator.', action='answer', question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a calculator. Here are examples:
            Q: What is 2+2?
            A: 4
            Q: What is 3+3?
            A: 6
            Now answer: What is 4+4?
            ''').strip()
        assert result == expected

    def test_format_missing_required_variable(self):
        '''Test format with missing required variable'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )
        with pytest.raises(KeyError, match='Missing required variables'):
            few_shot_template.format()

    def test_partial_variables_with_values(self):
        '''Test partial variables with fixed values'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={'role': 'assistant'}
        )

        result = few_shot_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_partial_variables_with_functions(self):
        '''Test partial variables with callable functions'''
        def get_role():
            return 'teacher'

        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={'role': get_role}
        )

        result = few_shot_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a teacher. Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_partial_variables_override_kwargs(self):
        '''Test that partial variables override kwargs values'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={'role': 'assistant'}
        )

        # Even if we pass role in kwargs, partial_vars should override
        result = few_shot_template.format(question='What is 4+4?', role='student')
        expected = textwrap.dedent('''
            You are a assistant. Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_partial_method(self):
        '''Test partial method to create new template'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['role', 'question'],
            partial_vars={}
        )

        partial_template = few_shot_template.partial(role='assistant')
        assert partial_template.prefix == 'You are a {role}. Examples:'
        assert partial_template.suffix == 'Answer: {question}'
        assert partial_template.required_vars == ['question']
        assert partial_template.partial_vars == {'role': 'assistant'}

        result = partial_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_partial_method_with_multiple_variables(self):
        '''Test partial method with multiple variables'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. {instruction} Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['role', 'instruction', 'question'],
            partial_vars={}
        )
        partial_template = few_shot_template.partial(role='assistant', instruction='Please follow these')
        assert partial_template.required_vars == ['question']
        assert partial_template.partial_vars == {'role': 'assistant', 'instruction': 'Please follow these'}

        result = partial_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Please follow these Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_partial_method_invalid_variable(self):
        '''Test partial method with invalid variable'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )
        with pytest.raises(KeyError, match='Variables not found in template'):
            few_shot_template.partial(invalid_var='value')

    def test_get_all_variables(self):
        '''Test get_all_variables method'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['role', 'question'],
            partial_vars={}
        )
        variables = few_shot_template._get_all_variables()
        assert set(variables) == {'role', 'question'}

    def test_get_all_variables_no_variables(self):
        '''Test get_all_variables with no variables'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='Examples:',
            suffix='Answer:',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=[],
            partial_vars={}
        )
        variables = few_shot_template._get_all_variables()
        assert not variables

    def test_validation_partial_vars_not_in_template(self):
        '''Test validation when partial_vars contains variables not in template'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError, match='partial_vars contains variables not found in template'):
            FewShotPromptTemplate(
                prefix='Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?', 'output': '4'}],
                egs_prompt_template=egs_template,
                required_vars=['question'],
                partial_vars={'invalid_var': 'value'}
            )

    def test_validation_required_vars_and_partial_vars_overlap(self):
        '''Test validation when required_vars and partial_vars have overlap'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError, match='required_vars and partial_vars have overlap'):
            FewShotPromptTemplate(
                prefix='You are a {role}. Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?', 'output': '4'}],
                egs_prompt_template=egs_template,
                required_vars=['role', 'question'],
                partial_vars={'role': 'assistant'}
            )

    def test_validation_missing_variables(self):
        '''Test validation when variables are missing from required_vars or partial_vars'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError, match='Missing variables in required_vars or partial_vars'):
            FewShotPromptTemplate(
                prefix='You are a {role}. Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?', 'output': '4'}],
                egs_prompt_template=egs_template,
                required_vars=['question'],
                partial_vars={}
            )

    def test_validation_extra_variables(self):
        '''Test validation when extra variables are provided'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError, match='Extra variables not found in template'):
            FewShotPromptTemplate(
                prefix='Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?', 'output': '4'}],
                egs_prompt_template=egs_template,
                required_vars=['question', 'extra_var'],
                partial_vars={}
            )

    def test_validation_example_compatibility(self):
        '''Test validation when examples are not compatible with egs_prompt_template'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError):
            FewShotPromptTemplate(
                prefix='Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?'}],  # Missing "output"
                egs_prompt_template=egs_template,
                required_vars=['question'],
                partial_vars={}
            )

    def test_validation_multiple_examples_compatibility(self):
        '''Test validation when multiple examples have compatibility issues'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')

        with pytest.raises(ValueError):
            FewShotPromptTemplate(
                prefix='Examples:',
                suffix='Answer: {question}',
                examples=[
                    {'input': 'What is 2+2?', 'output': '4'},  # Valid
                    {'input': 'What is 3+3?'}  # Missing "output"
                ],
                egs_prompt_template=egs_template,
                required_vars=['question'],
                partial_vars={}
            )

    def test_partial_function_error_handling(self):
        '''Test error handling when partial function raises exception'''
        def error_function():
            raise RuntimeError('Test error')

        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={'role': error_function}
        )

        with pytest.raises(TypeError, match='Error applying partial function for variable "role"'):
            few_shot_template.format(question='What is 4+4?')

    def test_example_formatting_error(self):
        '''Test error handling when example formatting fails'''
        with pytest.raises(ValueError, match='Example 0 missing required variables'):
            egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}\nExtra: {missing_var}')
            few_shot_template = FewShotPromptTemplate(
                prefix='Examples:',
                suffix='Answer: {question}',
                examples=[{'input': 'What is 2+2?', 'output': '4'}],
                egs_prompt_template=egs_template,
                required_vars=['question'],
                partial_vars={}
            )
            few_shot_template.format(question='What is 4+4?')

    def test_empty_examples(self):
        '''Test template with empty examples list'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a helpful assistant.',
            suffix='Please answer: {question}',
            examples=[],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )

        result = few_shot_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a helpful assistant.

            Please answer: What is 4+4?
            ''').strip()
        assert result == expected

    def test_complex_template(self):
        '''Test complex template with multiple variables and partial functions'''
        def get_timestamp():
            return '2024-01-01'

        def get_version():
            return '1.0.0'

        egs_template = PromptTemplate.from_template('Q: {question}\nA: {answer}')
        few_shot_template = FewShotPromptTemplate(
            prefix='System: {system}\nTimestamp: {timestamp}\nVersion: {version}\nExamples:',
            suffix='User: {user}\nPlease answer: {question}',
            examples=[
                {'question': 'What is 2+2?', 'answer': '4'},
                {'question': 'What is 3+3?', 'answer': '6'}
            ],
            egs_prompt_template=egs_template,
            required_vars=['system', 'user', 'question'],
            partial_vars={'timestamp': get_timestamp, 'version': get_version}
        )

        result = few_shot_template.format(system='Assistant', user='Hello', question='What is 4+4?')
        expected = textwrap.dedent('''
            System: Assistant
            Timestamp: 2024-01-01
            Version: 1.0.0
            Examples:
            Q: What is 2+2?
            A: 4
            Q: What is 3+3?
            A: 6
            User: Hello
            Please answer: What is 4+4?''').strip()
        assert result == expected

    def test_nested_partial_templates(self):
        '''Test creating nested partial templates'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. {instruction} Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['role', 'instruction', 'question'],
            partial_vars={}
        )

        # First partial
        partial1 = few_shot_template.partial(role='assistant')
        assert set(partial1.required_vars) == {'instruction', 'question'}
        assert partial1.partial_vars == {'role': 'assistant'}

        # Second partial
        partial2 = partial1.partial(instruction='Please follow these')
        assert partial2.required_vars == ['question']
        assert partial2.partial_vars == {'role': 'assistant', 'instruction': 'Please follow these'}

        result = partial2.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Please follow these Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_template_with_special_characters(self):
        '''Test template with special characters in variable names'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {user_role}. Examples:',
            suffix='Answer: {user_question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['user_role', 'user_question'],
            partial_vars={}
        )

        result = few_shot_template.format(user_role='assistant', user_question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_lambda_functions_in_partial_vars(self):
        '''Test lambda functions in partial variables'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        few_shot_template = FewShotPromptTemplate(
            prefix='You are a {role}. Count: {count} Examples:',
            suffix='Answer: {question}',
            examples=[{'input': 'What is 2+2?', 'output': '4'}],
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={'role': 'assistant', 'count': lambda: 42}
        )

        result = few_shot_template.format(question='What is 4+4?')
        expected = textwrap.dedent('''
            You are a assistant. Count: 42 Examples:
            Input: What is 2+2?
            Output: 4
            Answer: What is 4+4?''').strip()
        assert result == expected

    def test_examples_copy_in_partial(self):
        '''Test that examples are properly copied in partial method'''
        egs_template = PromptTemplate.from_template('Input: {input}\nOutput: {output}')
        original_examples = [{'input': 'What is 2+2?', 'output': '4'}]

        few_shot_template = FewShotPromptTemplate(
            prefix='Examples:',
            suffix='Answer: {question}',
            examples=original_examples,
            egs_prompt_template=egs_template,
            required_vars=['question'],
            partial_vars={}
        )

        partial_template = few_shot_template.partial()
        # Modify original examples
        original_examples.append({'input': 'What is 3+3?', 'output': '6'})

        # Partial template should not be affected
        assert len(partial_template.examples) == 1
        assert partial_template.examples == [{'input': 'What is 2+2?', 'output': '4'}]

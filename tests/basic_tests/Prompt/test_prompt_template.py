import pytest
from lazyllm.prompt_templates import PromptTemplate


class TestPromptTemplate:

    def test_basic_creation(self):
        prompt_template = PromptTemplate(
            template='Hello {name}, you are {age} years old',
            required_vars=['name', 'age'],
            partial_vars={}
        )
        assert prompt_template.template == 'Hello {name}, you are {age} years old'
        assert prompt_template.required_vars == ['name', 'age']
        assert prompt_template.partial_vars == {}

    def test_from_template_class_method(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old')
        assert prompt_template.template == 'Hello {name}, you are {age} years old'
        assert set(prompt_template.required_vars) == {'name', 'age'}
        assert prompt_template.partial_vars == {}

    def test_from_template_with_no_variables(self):
        prompt_template = PromptTemplate.from_template('Hello world')
        assert prompt_template.template == 'Hello world'
        assert prompt_template.required_vars == []
        assert prompt_template.partial_vars == {}

    def test_from_template_with_duplicate_variables(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, {name} is your name')
        assert prompt_template.template == 'Hello {name}, {name} is your name'
        assert prompt_template.required_vars == ['name']  # Should be deduplicated
        assert prompt_template.partial_vars == {}

    def test_format_basic(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old')
        result = prompt_template.format(name='Alice', age=25)
        assert result == 'Hello Alice, you are 25 years old'

    def test_format_missing_required_variable(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old')
        with pytest.raises(KeyError, match='Missing required variables'):
            prompt_template.format(name='Alice')

    def test_format_extra_variables(self):
        prompt_template = PromptTemplate.from_template('Hello {name}')
        result = prompt_template.format(name='Alice', extra='ignored')
        assert result == 'Hello Alice'

    def test_partial_variables_with_values(self):
        prompt_template = PromptTemplate(
            template='Hello {name}, you are {age} years old',
            required_vars=['name'],
            partial_vars={'age': 25}
        )
        result = prompt_template.format(name='Alice')
        assert result == 'Hello Alice, you are 25 years old'

    def test_partial_variables_with_functions(self):
        def get_year():
            return '2025'

        prompt_template = PromptTemplate(
            template='Hello {name}, this year is {year}',
            required_vars=['name'],
            partial_vars={'year': get_year}
        )
        result = prompt_template.format(name='Alice')
        assert result == 'Hello Alice, this year is 2025'

    def test_partial_variables_override_kwargs(self):
        prompt_template = PromptTemplate(
            template='Hello {name}, you are {age} years old',
            required_vars=['name'],
            partial_vars={'age': 25}
        )
        # Even if we pass age in kwargs, partial_vars should override
        result = prompt_template.format(name='Alice', age=30)
        assert result == 'Hello Alice, you are 25 years old'

    def test_partial_method(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old')
        partial_template = prompt_template.partial(age=25)

        assert partial_template.template == 'Hello {name}, you are {age} years old'
        assert partial_template.required_vars == ['name']
        assert partial_template.partial_vars == {'age': 25}

        result = partial_template.format(name='Alice')
        assert result == 'Hello Alice, you are 25 years old'

    def test_partial_method_with_multiple_variables(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old and live in {city}')
        partial_template = prompt_template.partial(age=25, city='New York')

        assert partial_template.required_vars == ['name']
        assert partial_template.partial_vars == {'age': 25, 'city': 'New York'}

        result = partial_template.format(name='Alice')
        assert result == 'Hello Alice, you are 25 years old and live in New York'

    def test_partial_method_invalid_variable(self):
        prompt_template = PromptTemplate.from_template('Hello {name}')
        with pytest.raises(KeyError, match='Variables not found in template'):
            prompt_template.partial(invalid_var='value')

    def test_validation_partial_vars_not_in_template(self):
        with pytest.raises(ValueError, match='partial_vars contains variables not found in template'):
            PromptTemplate(
                template='Hello {name}',
                required_vars=['name'],
                partial_vars={'invalid_var': 'value'}
            )

    def test_validation_required_vars_and_partial_vars_overlap(self):
        with pytest.raises(ValueError, match='required_vars and partial_vars have overlap'):
            PromptTemplate(
                template='Hello {name}, you are {age} years old',
                required_vars=['name', 'age'],
                partial_vars={'age': 25}
            )

    def test_validation_missing_variables(self):
        with pytest.raises(ValueError, match='Missing variables in required_vars or partial_vars'):
            PromptTemplate(
                template='Hello {name}, you are {age} years old',
                required_vars=['name'],
                partial_vars={}
            )

    def test_validation_extra_variables(self):
        with pytest.raises(ValueError, match='Extra variables not found in template'):
            PromptTemplate(
                template='Hello {name}',
                required_vars=['name', 'age'],
                partial_vars={}
            )

    def test_partial_function_error_handling(self):
        def error_function():
            raise RuntimeError('Test error')

        prompt_template = PromptTemplate(
            template='Hello {name}, you are {age} years old',
            required_vars=['name'],
            partial_vars={'age': error_function}
        )

        with pytest.raises(TypeError, match='Error applying partial function for variable "age"'):
            prompt_template.format(name='Alice')

    def test_format_template_variable_not_found(self):
        with pytest.raises(ValueError, match='Missing variables in'):
            prompt_template = PromptTemplate(
                template='Hello {name}',
                required_vars=[],
                partial_vars={}
            )

        with pytest.raises(KeyError, match='Missing required variables'):
            prompt_template = PromptTemplate.from_template(template='Hello {name}')
            prompt_template.format()

    def test_complex_template(self):
        def get_timestamp():
            return '2024-01-01'

        def get_version():
            return '1.0.0'

        prompt_template = PromptTemplate(
            template='System: {system}\nUser: {user}\nTimestamp: {timestamp}\nVersion: {version}',
            required_vars=['system', 'user'],
            partial_vars={'timestamp': get_timestamp, 'version': get_version}
        )

        result = prompt_template.format(system='Assistant', user='Hello')
        expected = 'System: Assistant\nUser: Hello\nTimestamp: 2024-01-01\nVersion: 1.0.0'
        assert result == expected

    def test_nested_partial_templates(self):
        prompt_template = PromptTemplate.from_template('Hello {name}, you are {age} years old and live in {city}')

        # First partial
        partial1 = prompt_template.partial(age=25)
        assert set(partial1.required_vars) == {'name', 'city'}
        assert partial1.partial_vars == {'age': 25}

        # Second partial
        partial2 = partial1.partial(city='New York')
        assert partial2.required_vars == ['name']
        assert partial2.partial_vars == {'age': 25, 'city': 'New York'}

        result = partial2.format(name='Alice')
        assert result == 'Hello Alice, you are 25 years old and live in New York'

    def test_template_with_special_characters(self):
        prompt_template = PromptTemplate.from_template('Hello {user_name}, your ID is {user_id_123}')
        result = prompt_template.format(user_name='Alice', user_id_123='12345')
        assert result == 'Hello Alice, your ID is 12345'

    def test_empty_template(self):
        prompt_template = PromptTemplate.from_template('')
        assert prompt_template.template == ''
        assert prompt_template.required_vars == []
        assert prompt_template.partial_vars == {}

        result = prompt_template.format()
        assert result == ''

    def test_template_with_curly_braces_not_variables(self):
        with pytest.raises(ValueError, match='Error getting template variables'):
            PromptTemplate.from_template('Hello {name}, this is not a variable: {not_a_var')

    def test_lambda_functions_in_partial_vars(self):
        prompt_template = PromptTemplate(
            template='Hello {name}, count: {count}',
            required_vars=['name'],
            partial_vars={'count': lambda: 42}
        )

        result = prompt_template.format(name='Alice')
        assert result == 'Hello Alice, count: 42'

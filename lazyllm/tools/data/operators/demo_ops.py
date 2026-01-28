from ..base_data import data_register


Demo1 = data_register.new_group('Demo1')
Demo2 = data_register.new_group('Demo2')

@data_register('data.Demo1', rewrite_func='forward_batch_input')
def build_pre_suffix(data, input_key='content', prefix='', suffix=''):
    assert isinstance(data, list)
    for item in data:
        item[input_key] = f'{prefix}{item.get(input_key, "")}{suffix}'
    return data

@data_register('data.Demo1', rewrite_func='forward')
def process_uppercase(data, input_key='content'):
    assert isinstance(data, dict)
    data[input_key] = data.get(input_key, '').upper()
    return data

class AddSuffix(Demo2):
    def __init__(self, suffix, input_key='content', **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        data[self.input_key] = f'{data.get(self.input_key, "")}{self.suffix}'
        return data

@data_register('data.Demo2', rewrite_func='forward')
def rich_content(data, input_key='content'):
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    new_res = [data]
    for i in range(2):
        new_data = data.copy()
        new_data[input_key] = f'{content} - part {i+1}'
        new_res.append(new_data)
    return new_res

@data_register('data.Demo2', rewrite_func='forward')
def error_prone_op(data, input_key='content'):
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    if content == 'fail':
        raise ValueError('Intentional error for testing.')
    data[input_key] = f'Processed: {content}'
    return data

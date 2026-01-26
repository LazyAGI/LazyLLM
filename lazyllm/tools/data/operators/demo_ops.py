from ..base_data import DataOperatorRegistry as register


demo2 = register.new_group('demo2')

@register('data.demo1', rewrite_func='forward_batch_input')
def build_pre_suffix(data, input_key='content', prefix='', suffix=''):
    assert isinstance(data, list)
    for item in data:
        item[input_key] = f'{prefix}{item.get(input_key, "")}{suffix}'
    return data

@register('data.demo1', rewrite_func='forward')
def process_uppercase(data, input_key='content'):
    assert isinstance(data, dict)
    data[input_key] = data.get(input_key, '').upper()
    return data

class AddSuffix(demo2):
    def __init__(self, suffix, input_key='content', **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        data[self.input_key] = f'{data.get(self.input_key, "")}{self.suffix}'
        return data

@register('data.demo2', rewrite_func='forward')
def rich_content(data, input_key='content'):
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    new_res = [data]
    for i in range(2):
        new_data = data.copy()
        new_data[input_key] = f'{content} - part {i+1}'
        new_res.append(new_data)
    return new_res

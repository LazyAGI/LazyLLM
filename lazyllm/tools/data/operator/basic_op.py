from ..base_data import DataOperatorRegistry


@DataOperatorRegistry.register(one_item=False, tag='all')
def build_pre_suffix(data, input_key='content', prefix='', suffix=''):
    assert isinstance(data, list)
    for item in data:
        item[input_key] = f'{prefix}{item.get(input_key, "")}{suffix}'
    return data

@DataOperatorRegistry.register
def process_uppercase(data, input_key='content'):
    assert isinstance(data, dict)
    data[input_key] = data.get(input_key, '').upper()
    return data

@DataOperatorRegistry.register
class AddSuffix:
    def __init__(self, suffix, input_key='content'):
        self.suffix = suffix
        self.input_key = input_key

    def __call__(self, data):
        assert isinstance(data, dict)
        data[self.input_key] = f'{data.get(self.input_key, "")}{self.suffix}'
        return data

@DataOperatorRegistry.register
def rich_content(data, input_key='content'):
    assert isinstance(data, dict)
    content = data.get(input_key, '')
    new_res = [data]
    for i in range(2):
        new_data = data.copy()
        new_data[input_key] = f'{content} - part {i+1}'
        new_res.append(new_data)
    return new_res

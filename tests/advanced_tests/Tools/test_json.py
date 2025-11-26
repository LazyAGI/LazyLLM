import lazyllm
from lazyllm.tools.tools import JsonExtractor, JsonConcentrator

class TestJsonExtractor(object):
    def test_extract(self):
        m = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
        e = JsonExtractor(m, schema='{"name": "John", "age": 30, "city": "New York"}',
                          field_descriptions={'name': 'name of the person, type is string',
                                              'age': 'age of the person, type is integer',
                                              'city': 'city of the person, type is string'})
        r = e('张三今年30岁，来自北京；李四今年35岁，来自上海，王五今年41岁')
        assert r == [{'name': '张三', 'age': 30, 'city': '北京'},
                     {'name': '李四', 'age': 35, 'city': '上海'},
                     {'name': '王五', 'age': 41, 'city': None}]

class TestJsonConcentrator(object):
    def test_concentrate_reduce(self):
        m = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
        c = JsonConcentrator(m, schema='{"Ability": "str", "Needs": "str", "Location": "str"}',
                             extra_requirements='The values should be comma and space separated strings with origin '
                             'order, no extra words should be added. If key or value is missing, ignor it.')
        r = c([{'Ability': '唱歌', 'Needs': '话筒', 'Location': '北京'},
               {'Ability': '跳舞', 'Needs': '', 'Location': '上海'},
               {'Ability': '画画', 'Needs': '画笔', 'Location': '北京'}])
        assert r == {'Ability': '唱歌, 跳舞, 画画', 'Needs': '话筒, 画笔', 'Location': '北京, 上海'}

    def test_concentrate_distinct(self):
        m = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
        c = JsonConcentrator(m, schema='{"name": "John", "age": 30, "city": "New York"}',
                             mode=JsonConcentrator.Mode.DISTINCT)
        r = c([{'name': '张三', 'age': 30, 'city': '北京'},
               {'name': '李四', 'age': 35, 'city': '上海'},
               {'name': '王五', 'age': 41, 'city': None},
               {'name': '张三', 'age': 30, 'city': '首都北京'},
               {'name': '李四', 'age': 35, 'city': '魔都'},])
        assert r == [{'name': '张三', 'age': 30, 'city': '北京'},
                     {'name': '李四', 'age': 35, 'city': '上海'},
                     {'name': '王五', 'age': 41, 'city': None}]

import random
import time
import pytest
import threading

import lazyllm
from lazyllm.common import ArgsDict, compile_func
from lazyllm.common import once_wrapper
from lazyllm.components.formatter import lazyllm_merge_query, encode_query_with_filepaths, decode_query_with_filepaths


class TestCommon(object):

    def test_common_argsdict(self):

        my_ob = ArgsDict({'a': '1', 'b': '2'})
        my_ob.check_and_update(my_ob)
        expected_output = '--a="1" --b="2"'
        assert my_ob.parse_kwargs() == expected_output

    def test_common_bind(self):

        def exam(a, b, c):
            return [a, b, c]

        num_list = [random.randint(1, 10) for _ in range(3)]
        r1 = lazyllm.bind(exam, num_list[0], lazyllm._0, num_list[2])
        ret_list = r1(num_list[1])
        assert ret_list == num_list

    def test_encode_and_decode_and_merge_query_with_filepaths(self):
        # Test encode
        query = 'hi'
        path_list = ['a', 'b']
        encode = encode_query_with_filepaths(query, path_list)
        assert encode == '<lazyllm-query>{"query": "hi", "files": ["a", "b"]}'
        assert encode_query_with_filepaths(query) == 'hi'

        # Test decode
        decode = decode_query_with_filepaths(encode)
        assert isinstance(decode, dict)
        assert 'query' in decode and 'files' in decode
        assert decode['query'] == query
        assert decode['files'] == path_list
        assert decode_query_with_filepaths(query) == query

        # Test Merge
        assert lazyllm_merge_query(query) == query
        assert lazyllm_merge_query(query, query, query) == query * 3
        assert lazyllm_merge_query(query, encode) == '<lazyllm-query>{"query": "hihi", "files": ["a", "b"]}'
        assert lazyllm_merge_query(encode, encode) == ('<lazyllm-query>{"query": "hihi", "files": '
                                                       '["a", "b", "a", "b"]}')
        assert lazyllm_merge_query(encode, query, query) == ('<lazyllm-query>{"query": "hihihi", '
                                                             '"files": ["a", "b"]}')

    def test_common_cmd(self):

        ret = lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['a'])
        assert str(ret) == 'python a  --c=d'

        ret = lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['c'])
        assert str(ret) == 'python a --a=b '

        ret = lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['d'])
        assert str(ret) == 'python a --a=b --c=d'

    def test_common_timeout(self):
        from lazyllm.common.common import TimeoutException

        with pytest.raises(TimeoutException):
            with lazyllm.timeout(1, msg='hello'):
                time.sleep(2)

    def test_common_tread(self):

        def is_equal2(x):
            if x == 2:
                return x
            else:
                raise Exception

        ts = [lazyllm.Thread(target=is_equal2, args=(inp, )) for inp in [2, 3]]
        [t.start() for t in ts]

        assert ts[0].get_result() == 2
        with pytest.raises(Exception):
            ts[1].get_result()

    def test_common_makerepr(self):

        r1 = lazyllm.make_repr('a', 1)
        r2 = lazyllm.make_repr('b', 2)
        rr = lazyllm.make_repr('c', 3, subs=[r1, r2])
        assert rr == '<c type=3>\n |- <a type=1>\n └- <b type=2>\n'

    def test_compile_func(self):
        str1 = "def identity(v): return v"
        identity = compile_func(str1)
        assert identity("abc") == "abc"
        assert identity(12345) == 12345

        str2 = "def square(v): return v * v"
        square = compile_func(str2)
        assert square(3) == 9
        assert square(18) == 324


class TestCommonOnce(object):

    @once_wrapper
    def once_func(self):
        self._count += 1

    @once_wrapper
    def once_func_with_exception(self):
        self._count += 1
        raise RuntimeError('once exception')

    def test_callonce(self):
        self._count = 0
        assert not self.once_func.flag
        self.once_func()
        assert self._count == 1
        assert self.once_func.flag
        self.once_func()
        assert self._count == 1

    def test_callonce_exception(self):
        self._count = 0
        assert not self.once_func_with_exception.flag
        with pytest.raises(RuntimeError, match='once exception'):
            self.once_func_with_exception()
        assert self._count == 1
        assert self.once_func_with_exception.flag
        with pytest.raises(RuntimeError, match='once exception'):
            self.once_func_with_exception()
        assert self._count == 1


class TestCommonGlobals(object):

    def _lazyllm_worker(self):
        assert lazyllm.globals['a'] == 1
        assert lazyllm.globals['chat_history'] == {}
        assert lazyllm.globals['global_parameters']['key'] == 'value'

    def _normal_worker(self):
        assert 'a' not in lazyllm.globals
        assert lazyllm.globals._sid == f'tid-{hex(threading.get_ident())}'
        assert lazyllm.globals['chat_history'] == {}
        assert lazyllm.globals['global_parameters'] == {}

    def test_globals(self):
        assert lazyllm.globals._sid == f'tid-{hex(threading.get_ident())}'
        assert lazyllm.globals['chat_history'] == {}
        assert lazyllm.globals['global_parameters'] == {}
        lazyllm.globals['global_parameters']['key'] = 'value'
        t = lazyllm.Thread(target=self._lazyllm_worker)
        t.start()
        t.join()
        t = threading.Thread(target=self._normal_worker)
        t.start()
        t.join()


class TestCommonRegistry(object):
    def test_component_registry(self):
        lazyllm.component_register.new_group('mygroup')

        @lazyllm.component_register('mygroup')
        def myfunc(input):
            return input

        assert lazyllm.mygroup.myfunc()(1) == 1
        assert lazyllm.mygroup.myfunc(launcher=lazyllm.launchers.empty)(1) == 1

        lazyllm.mygroup.remove('myfunc')
        with pytest.raises(AttributeError):
            lazyllm.mygroup.myfunc()(1)

        @lazyllm.component_register('mygroup.subgroup')
        def myfunc2(input):
            return input

        assert lazyllm.mygroup.subgroup.myfunc2()(1) == 1
        assert lazyllm.mygroup.subgroup.myfunc2(launcher=lazyllm.launchers.empty)(1) == 1

    def test_custom_registry(self):
        class CustomClass(object, metaclass=lazyllm.common.registry.LazyLLMRegisterMetaClass):
            def __call__(self, a, b):
                return self.forward(a + 1, b * 2)

            def forward(self, a, b):
                raise NotImplementedError('forward is not implemented')

        reg = lazyllm.Register(CustomClass, 'forward')
        reg.new_group('custom')

        @reg('custom')
        def test(a, b): return a + b

        @reg.forward('custom')
        def test2(a, b): return a * b

        assert lazyllm.custom.test()(1, 2) == 6
        assert lazyllm.custom.test2()(1, 2) == 8

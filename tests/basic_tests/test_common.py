import lazyllm
from lazyllm.common import ArgsDict
import random
import time
import pytest

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

    def test_common_llmreqreshelper(self):

        h = lazyllm.ReqResHelper()
        assert h.make_request(1, a=3, b=2)
        assert h.make_request(1, 2, a=3)

        r1 = lazyllm.LazyLlmResponse(messages=1, trace='t1')
        r2 = lazyllm.LazyLlmResponse(messages=2, trace='t2')
        assert h.make_request(r1)
        assert h.make_request(r2)
        assert h.trace == 't1t2'

        assert h.make_response('abc')
        assert h.trace == 't1t2'

        r3 = lazyllm.LazyLlmResponse(messages=3, trace='t3')
        assert h.make_response(r3)
        assert h.trace == 't1t2'

    def test_common_makerepr(self):

        r1 = lazyllm.make_repr('a', 1)
        r2 = lazyllm.make_repr('b', 2)
        rr = lazyllm.make_repr('c', 3, subs=[r1, r2])
        assert rr == '<c type=3>\n |- <a type=1>\n └- <b type=2>\n'

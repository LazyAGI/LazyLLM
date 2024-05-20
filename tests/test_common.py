import lazyllm
from lazyllm.common import ArgsDict
import random
import time, pytest

class TestFn_Common(object):
    
    def test_common_argsdict(self):
        
        my_ob = ArgsDict({'a':'1', 'b':'2'})
        my_ob.check_and_update(my_ob) 
        # print(my_ob)
        expected_output = '--a="1" --b="2"'
        assert my_ob.parse_kwargs() == expected_output
        
    def test_common_bind(self):
        
        def exam(a, b ,c):
            return [a, b, c]
        
        num_list = [random.randint(1, 10) for _ in range(3)]
        r1 = lazyllm.bind(exam, num_list[0], lazyllm._0, num_list[2])
        ret_list = r1(num_list[1])
        assert ret_list == num_list
        
    def test_common_cmd(self):
        
        ret =lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['a'])
        assert str(ret) == 'python a  --c=d'
        
        ret =lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['c'])
        assert str(ret) == 'python a --a=b '
        
        ret =lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['d'])
        assert str(ret) == 'python a --a=b --c=d'
        
    def test_common_timeout(self):
        
        with pytest.raises(TimeoutError) as e:
            with lazyllm.timeout(1, msg='hello'):
                time.sleep(2)
                
    def test_common_tread(self):
        # TODO: 
        pass
    
    def test_common_llmrequest(self):
        # TODO: 
        pass
    
    def test_common_llmresponse(self):
        # TODO: 
        pass
    
    def test_common_llmreqreshelper(self):
        # TODO: 
        pass
    
    def test_common_makerepr(self):
        
        r1 =lazyllm.make_repr('a', 1)
        r2 =lazyllm.make_repr('b', 2)
        rr = lazyllm.make_repr('c', 3, subs=[r1, r2])
        assert rr == '<c type=3>\n |- <a type=1>\n └- <b type=2>\n'
        

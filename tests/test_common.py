import lazyllm
from lazyllm.common import ArgsDict
import random
import time, pytest

class TestFn_Common(object):
    
    def test_common_argsdict(self):
        
        my_ob = ArgsDict({'a':'1', 'b':'2'})
        my_ob.check_and_update(my_ob) #update了啥
        # print(my_ob)
        assert my_ob.parse_kwargs() == '--a="1" --b="2"'
        
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
        assert str(ret) == 'python a --a=b'
        
        ret =lazyllm.LazyLLMCMD('python a --a=b --c=d', no_displays=['d'])
        assert str(ret) == 'python a --a=b --c=d'
        
    def test_common_timeout(self):
        
        with pytest.raises(TimeoutError) as e:
            with lazyllm.timeout(1, msg='hello'):
                time.sleep(2)
                
    
        
            
    
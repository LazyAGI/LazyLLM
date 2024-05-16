import lazyllm
from lazyllm.configs import Mode
import os


class TestFn_Config(object):
    def test_config_mode(self):
        print(os.environ.get('LAZYLLM_DISPLAY'))
        assert lazyllm.config['mode'] == Mode.Normal
    
    # def test_config_disp(self): 
    #     os.environ['LAZYLLM_DISPLAY']='1'
    #     print(os.environ.get('LAZYLLM_DISPLAY'))
    #     ret = lazyllm.config['mode']
    #     print(ret)  #Mode.Normal
        #assert ret == Mode.Display
        

        
        

        
        
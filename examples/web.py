import lazyllm
from lazyllm import ModuleResponse

class Test(object):
    def __init__(self):
        self._stream_output = True

    @property
    def stream_output(self):
        return self._stream_output
    
    @stream_output.setter
    def stream_output(self, b):
        self._stream_output = b

    def __call__(self, input):
        if self._stream_output:
            return test_stream(input)
        else:
            return test(input)

def test_stream(a):
    for i in range(50):
        yield ModuleResponse(messages=f'[{a}-{i}]', trace=f'calced{a} idx={i}!')
        import time
        time.sleep(0.1)

def test(a):
    return ModuleResponse(messages=f'in{a}!!', trace=f'calced{a}!')

m = lazyllm.WebModule(Test(), stream_output=True)
m.update()
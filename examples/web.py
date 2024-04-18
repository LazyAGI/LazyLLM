import lazyllm
from lazyllm import ModuleResponse

m = lazyllm.ActionModule(
        [lazyllm.ActionModule((lambda x:x), return_trace=True),
         lazyllm.TrainableModule(stream=True).finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune').prompt('hello world <{input}>')])
m2 = lazyllm.ServerModule(m, stream=True)
m3 = lazyllm.WebModule(m2, trace_mode=lazyllm.WebModule.TraceMode.Appendix)
#m3 = lazyllm.WebModule(m2, components=[('do_sample', 'Checkbox', True), ('temperature', 'Text', 0.1)])
m3.update()
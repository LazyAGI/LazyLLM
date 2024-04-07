import lazyllm
from lazyllm import ModuleResponse

m = lazyllm.TrainableModule(stream=True).finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune')
m2 = lazyllm.ServerModule(m, stream=True)
m3 = lazyllm.WebModule(m2)
m3.update()
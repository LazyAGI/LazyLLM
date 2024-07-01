import lazyllm
from lazyllm import finetune, deploy
import time

t1 = lazyllm.TrainableModule(stream=False).finetune_method(finetune.dummy).deploy_method(
    deploy.dummy).mode('finetune').prompt('hello world1 <{input}>')
prompter = lazyllm.Prompter(prompt='hello world2 <{input}>, hisory is <{history}>',
                            history_symbol='history', eoh='[EOH]', eoa='[EOA]')
t2 = lazyllm.TrainableModule(stream=True).finetune_method(finetune.dummy).deploy_method(
    deploy.dummy).mode('finetune').prompt(prompter)
t1.name = 'M1'
t2.name = 'M2'

s1 = lazyllm.ServerModule(t1, return_trace=True)
s2 = lazyllm.ServerModule(lazyllm.ActionModule(
        [lazyllm.ActionModule((lambda x:f'{x}~~~~'), return_trace=True), t2]), stream=True)
s1.name = 'S1'
s2.name = 'S2'
s0 = lazyllm.ServerModule(lazyllm.pipeline(s1, s2), stream=True)

w = lazyllm.WebModule(s0, port=[20570, 20571, 20572], components={
        t1:[('do_sample', 'Checkbox', True), ('temperature', 'Text', 0.1)],
        t2:[('do_sample', 'Checkbox', False), ('temperature', 'Text', 0.2)]},
    history=[t2])

t1 = lazyllm.ForkProcess(target=w.update, args=(), sync=False)
t2 = lazyllm.ForkProcess(target=w.update, args=(), sync=False)
t1.start()
time.sleep(2)
t2.start()
t1.join()
t2.join()
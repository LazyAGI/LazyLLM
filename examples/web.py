import lazyllm
import time

t1 = lazyllm.TrainableModule(stream=False).finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune').prompt('hello world1 <{input}>')
t2 = lazyllm.TrainableModule(stream=True).finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune').prompt('hello world2 <{input}>')
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
    t2:[('do_sample', 'Checkbox', False), ('temperature', 'Text', 0.2)]})
import multiprocessing
t1 = multiprocessing.Process(target=w.update, args=())
t2 = multiprocessing.Process(target=w.update, args=())
t1.start()
time.sleep(1)
t2.start()
t1.join()
t2.join()
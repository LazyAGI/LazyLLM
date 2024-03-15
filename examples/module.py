import lazyllm

m = lazyllm.TrainableModule('b', 't').finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune')
m2 = lazyllm.ServerModule(m)
m3 = lazyllm.ServerModule(m2)
m4 = lazyllm.ServerModule(m3)
m4.evalset([1, 2, 3, 4, 5, 6])
m4.update()
print(m4.eval_result)
r = m4(7)
print(r)
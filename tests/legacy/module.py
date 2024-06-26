import lazyllm

m = lazyllm.TrainableModule(lazyllm.Option(['b1', 'b2', 'b3']), 't'
        ).finetune_method(finetune.dummy, **dict(a=lazyllm.Option(['f1', 'f2']))
        ).deploy_method(deploy.dummy).mode('finetune')
m2 = lazyllm.ServerModule(m, post=lambda x, *, ori: f'post2({x})')
m3 = lazyllm.ServerModule(m2, post=lambda x, ori: f'post3({x})')
m4 = lazyllm.ServerModule(m3,
                          pre=lambda input, background='b': dict(inputs=input, background=f'{background}-1', appendix=1),
                          post=lambda x: f'post4({x})')
m4.prompt('m4: {input}')
m.prompt('m: i-{inputs}, b-{background}, a-{appendix}')
m4.evalset([1, 2, 3, 4, 5, 6])
m4.update()
print(m4.eval_result)
r = m4(dict(input=8, background=900))
print(r)

c = lazyllm.ResultCollector()

print(m4.__class__.__name__)
ppl = lazyllm.pipeline(
    gendata=lambda x: dict(input=x, background='back'),
    coll1=c('data'),
    module=m4,
    coll2=c('result'),
    getr=lambda *args, **kw: c, 
)
print(ppl.module.__class__.__name__)
print(ppl('inp'))

print('-----------------------------')

m5 = lazyllm.TrialModule(m4)
m5.update()
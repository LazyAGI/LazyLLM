import lazyllm

m = lazyllm.TrainableModule('b', 't').finetune(finetune.dummy).deploy(deploy.dummy).mode(
    'finetune')
m2 = lazyllm.ServerModule(m, post=lambda x, *, ori: f'post2({x})')
m3 = lazyllm.ServerModule(m2, post=lambda x, ori: f'post3({x})')
m4 = lazyllm.ServerModule(m3,
                          pre=lambda input, background='b': dict(input=input, background=background, appendix=1),
                          post=lambda x: f'post4({x})')
m4.prompt('m4: {input}', response_split=None)
m4.evalset([1, 2, 3, 4, 5, 6])
m4.update()
print(m4.eval_result)
m4.prompt('m4: {input}, {background}, {appendix}', response_split=None)
r = m4(input=8, background=900)
print(r)

c = lazyllm.ResultCollector()

print(m4.__class__.__name__)
ppl = lazyllm.pipeline(
    gendata=lambda x: lazyllm.kwargs(input=x, background='back'),
    coll1=c('data'),
    module=m4,
    coll2=c('result'),
    getr=lambda *args, **kw: c, 
)
print(ppl.module.__class__.__name__)
print(ppl.start('inp').result)
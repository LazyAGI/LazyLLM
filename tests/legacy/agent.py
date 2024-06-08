import lazyllm
from lazyllm import pipeline, parallel, diverter, switch, loop, ifs, Identity

def is_text(input):
    return 'text' in input 

def is_photo(input):
    return 'photo' in input

def is_voice(input):
    return 'voice' in input

def ptt(input):
    assert 'photo' in input
    return input.replace('photo', 'pttext')

def vtt(input):
    assert 'voice' in input
    return input.replace('voice', 'vttext')

def combine(context, input):
    return f'{context}<eos>{input}'

def minganci(input):
    return 'mgc' in input

def planner(input):
    return f'<plan> {input}'

def act1(x):
    return 'act1' in x

executor = switch({
    act1 : (lambda x: '<ActA-done>' + x),
    (lambda x: 'act2' in x) : (lambda x: '<ActB-done>' + x),
    (lambda x: 'act3' in x) : (lambda x: '<ActC-done>' + x),
    'default' : (lambda x: '<No-action>' + x),
})
    
def chat(input):
    return input

duomotai = diverter(
    Identity,
    switch({
        is_text: Identity,
        is_photo: ptt,
        is_voice: vtt,
        'default': lambda x: 'invalid input'
    })
)
 
# (ctx, input)
m = lazyllm.ActionModule(
    duomotai,
    combine,
    ifs(minganci, Identity, loop(
            planner,
            lazyllm.ActionModule(executor),
            count=1
        )
    ),
    chat
)

print(m('ctx', 'text-act1'))
print(m('ctx', 'photo-act2'))
print(m('ctx', 'voice'))
print(m('mgc', 'photo'))
print(m.submodules)
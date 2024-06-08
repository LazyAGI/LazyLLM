import json

import lazyllm
from lazyllm import launchers, deploy

'''
Set the environment variable with: 
export LAZYLLM_SLURM_PART=pat_rd
'''
# ====================================== Setting parameters:

base_model = 'internlm2-chat-1_8b'
deploy_config = {
    'launcher': launchers.slurm(
        partition='pat_rd',
        nnode=1,
        nproc=1,
        ngpus=1,
        sync=False
        ),
}

# ====================================== Build Moudule:

m1 = lazyllm.TrainableModule(base_model, '').deploy_method(deploy.lightllm, **deploy_config)

def pre_func(kx:str):
    print("=== Before pre_func: ", kx)
    kx = f'<|im_start|>user\n{kx}<|im_end|>\n<|im_start|>assistant\n'
    print("=== After pre_func: ", kx)
    return kx

def post_func(x:str):
    print("=== Before post_func: ", x)
    x_splited = json.loads(x)['generated_text'][0].rsplit('<|im_start|>assistant\n', maxsplit=1)
    if len(x_splited) == 2:
        x = x_splited[1].strip()
    print("=== After post_func: ", x)
    return x

m = lazyllm.ServerModule(m1, pre=pre_func, post=post_func)

# ====================================== Add Eval dataset:

m.evalset(['介绍一下你自己', '李白和李清照是什么关系'])


# ====================================== Run:
m.update_server()

m.eval()
print(m.eval_result)

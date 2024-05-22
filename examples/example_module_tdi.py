
import lazyllm
from lazyllm import pipeline, launchers

'''
This demo shows the entire process of a module:
- finetune
- deploy
- eval
'''

# ====================================== Setting parameters:
# 1. prompt template:
template_stage1 = (
    'Below is an instruction that describes a task, '
    'paired with an input that provides further context. '
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n将下列复句按逻辑关系添加OR()或AND()表示，如果不是复句，保留原样\n\n'
    '### Input:\n{input}\n\n'
    '### Response:\n'
    )

# 2. data path:
stage1_data_path = '/mnt/lustrenew/share_data/sunxiaoye/Dataset/Finture_TDX/step1_0103_xuzhiguo.json'

# 3. finetune and deploy parameters:
base_model = '/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/ChatGLM3-6B_base'
target_path = '/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/lazy_tdx0329_2'
stage1_args = {
    'finetune': {
        'model_name':'chatglm3-6b',
        'gpu_count':8,
    },
    'deploy': {
        'launcher': launchers.slurm(
            partition='pat_rd',
            nnode=1,
            nproc=1,
            ngpus=1,
            sync=False
            ),
    }
}

# ====================================== Build Moudule:

m1 = lazyllm.TrainableModule(base_model, target_path
        ).finetune_method(finetune.AutoFinetune, **stage1_args['finetune']
        ).deploy_method(deploy.lightllm, **stage1_args['deploy']
        ).mode('finetune').trainset(stage1_data_path).prompt(template_stage1)

def combine_json(*args):
    print("=== All-Out: ", args)
    return f'{args}'

def pre_func(kx):
    print("=== Pre-Func: ", kx, type(kx))
    return kx

def after_func(x):
    print("=== After-Func: ", x, type(x))
    return f'After-Func------{x}'

m = lazyllm.ServerModule(lazyllm.ActionModule(
    pipeline(
        lazyllm.ServerModule(m1, pre=pre_func, post=after_func),
        combine_json,
        )
    ))

# ====================================== Add Eval dataset:

m.evalset(['hello! ', 'make dream come true', '再见'])


# ====================================== Run:
m.update()
print(m.eval_result)

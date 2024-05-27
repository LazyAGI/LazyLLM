import sys
sys.path.append('..')

import lazyllm
try:
   from builtins import package, dataproc, finetune, deploy, launchers, validate
except ImportError:
   from lazyllm import package, dataproc, finetune, deploy, launchers, validate

from lazyllm import root, bind, _0, _1

@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    return package(idx + 1, idx + 1)

@lazyllm.llmregister('Validate')
def eval_stage1(url):
    print(f'url {url} eval_stage1 done')

@lazyllm.llmregister('Validate')
def eval_stage2(url):
    print(f'url {url} eval_stage2 done')

@lazyllm.llmregister('validate')
def eval_all(evalset, url1, url2, job=None):
    print(f'eval all. evalset: {evalset}, url: {url1} and {url2} eval_all done. job: {job}')

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    lazyllm.parallel(
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1', launcher=launchers.slurm()),
            deploy.lightllm(),
            post_action=validate.eval_stage1,
        ),
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model2', target_path='./finetune-target2', launcher=launchers.slurm),
            deploy.lightllm(port=8080),
            post_action=validate.eval_stage1,
        ),
    ),
    bind(validate.eval_all, 'valset-1', _0, _1),
)

ppl(0)

print('---------------------------')
print('---------------------------')

named_ppl = lazyllm.pipeline(
    data=dataproc.gen_data(),
    finetune=lazyllm.parallel(
        stage1=lazyllm.pipeline(
            sft=finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1', launcher=launchers.slurm()),
            deploy=deploy.lightllm(),
            post_action=validate.eval_stage1,
        ),
        stage2=lazyllm.pipeline(
            sft=finetune.alpacalora(base_model='./base-model2', target_path='./finetune-target2', launcher=launchers.slurm),
            deploy=deploy.lightllm(port=8080),
            post_action=validate.eval_stage1,
        ),
    ),
    val=bind(validate.eval_all, root.finetune.stage2.post_action, _0, _1, root.finetune.stage1.deploy.job),
)

named_ppl(0)

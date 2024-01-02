import lazyllm
try:
   from builtins import package, dataproc, finetune, deploy, launchers, validate
except ImportError:
   from lazyllm import package, dataproc, finetune, deploy, launchers, validate

@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    return package([idx + 1, idx + 1])

@lazyllm.llmregister('deploy')
def mydeploy(idx):
    print(f'idx {idx}: deploy done')
    return idx + 1

@lazyllm.llmregister('Validate')
def eval_stage1(idx):
    print(f'idx {idx}: eval_stage1 done')
    return idx + 1

@lazyllm.llmregister('Validate')
def eval_stage2(idx):
    print(f'idx {idx}: eval_stage2 done')
    return idx + 1

@lazyllm.llmregister('validate')
def eval_all(launcher, idx):
    print(f'idx {idx}: eval_all done')
    return idx[0] + idx[1]

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    lazyllm.parallel(
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1', launcher=launchers.slurm()),
            deploy.lightllm()
        ),
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model2', target_path='./finetune-target2', launcher=launchers.slurm),
            deploy.lightllm()
        ),
    ),
)
ppl.run(0)
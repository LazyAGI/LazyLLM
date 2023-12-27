import lazyllm
try:
   from builtins import dataproc, finetune, deploy, launchers, validate
except ImportError:
   from lazyllm import dataproc, finetune, deploy, launchers, validate

@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    return idx + 1

@lazyllm.llmregister('finetune')
def myfinetune(idx):
    print(f'idx {idx}: finetune done')
    return idx + 1

@lazyllm.llmregister('finetune')
def mergeWeights(idx):
    print(f'idx {idx}: merge weights done')
    return idx + 1

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
            finetune.myfinetune(1, launcher=launchers.empty),
            finetune.mergeWeights(1),
            deploy(),
            post_action=lazyllm.pipeline(validate.eval_stage1()),
        ),
        lazyllm.pipeline(
            finetune.myfinetune(1),
            finetune.mergeWeights(1),
            deploy(),
            post_action=lazyllm.pipeline(validate.eval_stage2()),
        ),
    ),
    validate.eval_all()
)
ppl.run(0)
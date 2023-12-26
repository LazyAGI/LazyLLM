import lazyllm
try:
   from builtins import dataProcessing, finetune, deploy, launcher, validate
except ImportError:
   from lazyllm import dataProcessing, finetune, deploy, launcher, validate

ppl = lazyllm.pipeline(
    dataProcessing.gen_data(),
    lazyllm.parallel(
        lazyllm.pipeline(
            finetune.finetune(launcher=launcher.slurm()),
            finetune.mergeWeights(launcher=launcher.slurm()),
            deploy(launcher=launcher.slurm()),
            post_action=lazyllm.pipeline(validate.eval_stage1()),
        ),
        lazyllm.pipeline(
            finetune.finetune(),
            finetune.mergeWeights(launcher=launcher.slurm()),
            deploy(),
            post_action=lazyllm.pipeline(validate.eval_stage2()),
        ),
    ),
    validate.eval()
)
ppl.run()
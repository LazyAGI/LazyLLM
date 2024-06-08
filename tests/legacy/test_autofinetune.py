import lazyllm
from builtins import package, dataproc, finetune, launchers


@lazyllm.component_register('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    datapath = '/file/to/yourfile.json'
    return package(datapath, None)

finetuner = finetune.AutoFinetune(
        target_path='Internlm2-chat-20b/lazy_demo/lora',
        base_model='internlm2-chat-20b',
        batch_size=4,
        launcher=launchers.slurm(partition='pat_rd',
                                 ngpus=2,
                                 nproc=2,
                                 )
    )

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    finetuner,
)

ppl(0)

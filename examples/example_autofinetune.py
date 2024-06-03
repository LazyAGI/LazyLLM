import lazyllm
from builtins import package, dataproc, finetune, launchers


@lazyllm.component_register('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    datapath = '/mnt/lustrenew/share_data/sunxiaoye/Dataset/Finture_TDX/step1_0103_xuzhiguo.json'
    return package(datapath, None)

finetuner = finetune.AutoFinetune(
        target_path='/mnt/lustrenew/share_data/sunxiaoye/Models/Internlm2-chat-20b/lazy_demo/lora',
        base_model='/mnt/lustrenew/share_data/sunxiaoye/Models/internlm2-chat-20b',
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

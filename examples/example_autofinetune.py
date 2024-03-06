import lazyllm
from builtins import package, dataproc, finetune


@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    datapath = '/mnt/lustrenew/share_data/sunxiaoye/Dataset/Finture_TDX/step1_0103_xuzhiguo.json'
    return package(datapath, None)

finetuner = finetune.AutoFinetune(
        model_name='chatglm3-6b',
        gpu_count=4,
        target_path='/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/easy_tdx0304',
        base_model='/mnt/lustrenew/share_data/sunxiaoye/lazyllm_models_home/chatglm3-6b',
    )

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    finetuner,
)

ppl.start(0)

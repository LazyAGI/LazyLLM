import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lazyllm
try:
   from builtins import package, dataproc, finetune, deploy, launchers, validate
except ImportError:
   from lazyllm import package, dataproc, finetune, deploy, launchers, validate


@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    datapath = '/mnt/lustrenew/share_data/sunxiaoye/Dataset/Finture_TDX/step1_0103_xuzhiguo.json'
    return package(datapath, idx + 1)

def before_func(input_json):
    print("Before LLM: ", input_json)
    return input_json

def after_func(input_json, llm_output):
    print("After LLM: ", llm_output)
    llm_output = json.loads(llm_output.decode('utf-8'))["generated_text"][0]
    return json.dumps(llm_output)

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    finetune.alpacalora(
        base_model='/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/ChatGLM3-6B_base',
        target_path='/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/easy_tdx0103/Stage1',
        merge_path='/mnt/lustrenew/share_data/sunxiaoye/Models/ChatGLM3-6B/easy_tdx0103/Merge_Stage1',
        cp_files='tokeniz*',
        model_name='ChatGLM3-6B-S1',
        batch_size=64,
        micro_batch_size=4,
        num_epochs=2,
        learning_rate=5.e-4,
        cutoff_len=1030,
        filter_nums=1024,
        val_set_size=200,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules='[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]',
        modules_to_save='[word_embeddings, output_layer]',
        deepspeed='../lazyllm/llms/finetune/alpaca-lora/ds.json',
        prompt_template_name='alpaca',
        train_on_inputs=True,
        show_prompt=False,
        launcher=launchers.slurm(
            partition='pat_rd',
            nnode=2,
            nproc=16,
            ngpus=8,
            )
        ),
    deploy.lightllm(
        launcher=launchers.slurm(
            partition='pat_rd',
            nnode=1,
            nproc=1,
            ngpus=1,
            sync=False
            ),
    ),
)
ppl.start(0)

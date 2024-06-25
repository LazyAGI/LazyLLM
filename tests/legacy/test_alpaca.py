import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lazyllm
try:
   from builtins import package, dataproc, finetune, deploy, launchers, validate
except ImportError:
   from lazyllm import package, dataproc, finetune, deploy, launchers, validate


@lazyllm.component_register('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    datapath = 'alpaca/alpaca_data_zh_51k.json'
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
        base_model='models/internlm2-chat-7b',
        target_path='Internlm2-chat-7b/lazy_demo/lora',
        merge_path='Internlm2-chat-7b/lazy_demo/merge',
        cp_files='tokeniz*',
        model_name='internlm7b',
        batch_size=8,
        micro_batch_size=2,
        num_epochs=2,
        learning_rate=5.e-4,
        cutoff_len=1030,
        filter_nums=1024,
        val_set_size=200,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules='[wo,wqkv]',
        modules_to_save='[tok_embeddings,output]',
        prompt_template_name='alpaca',
        train_on_inputs=True,
        show_prompt=False,
        launcher=launchers.slurm(
            partition='pat_rd',
            nnode=1,
            nproc=4,
            ngpus=4,
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
ppl(0)

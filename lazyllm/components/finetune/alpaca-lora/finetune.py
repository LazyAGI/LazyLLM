# Copyright (c) Eric J. Wang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from alpaca-lora/finetune.py (https://github.com/tloen/alpaca-lora/)
# Copyright (c) 2023 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from typing import List
import subprocess

import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
import numpy as np
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from transformers import AutoTokenizer, AutoModel
import deepspeed

from utils.prompter import Prompter

def init_dist(port):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "SLURM_JOB_ID" in os.environ:
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = str(port)
        print(addr, port)
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        rank = proc_id
        world_size = ntasks

        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)

        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['SLURM_LOCALID'])

        print("comm: ", world_size, rank, local_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        torch.cuda.set_device(local_rank)
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        rank = os.getenv("RANK", None)
        if rank:
            rank = int(rank)
        else:
            rank = local_rank

    deepspeed.init_distributed()
    return rank

def train( # noqa C901
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "",
    output_dir: str = os.path.abspath("./output_dir"),
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    filter_nums: int = 512,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    modules_to_save: List[str] = None,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    deepspeed: str = None,  # deepspeed config file path.
    show_prompt: bool = False,
    nccl_port: int = 19080,
):
    rank = init_dist(nccl_port)
    if int(os.environ.get("RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"filter_nums: {filter_nums}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"modules_to_save: {modules_to_save}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"show prompt: {show_prompt}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter.from_template(prompt_template_name, show=show_prompt)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if rank == 0:
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ['WANDB_DISABLED'] = 'true'

    model = AutoModel.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) > cutoff_len:
            return None
        if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        output = data_point.pop("output", None)
        full_prompt = prompter.generate_prompt(data_point, label=output)
        tokenized_full_prompt = tokenize(full_prompt)
        if not tokenized_full_prompt:
            return None
        train_on_inputs = True
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point)
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    modules_to_save = list(modules_to_save) if modules_to_save else None
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    datas = []
    data_names = ''
    if os.path.isfile(data_path):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            datas.append(load_dataset("json", data_files=data_path, split='train'))
        else:
            datas.append(load_dataset(data_path))
    elif os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    aa = load_dataset("json", data_files=file_path, split='train')
                    key_list = aa.column_names
                    for key in key_list:
                        if key not in ['instruction', 'output', 'input']:
                            aa = aa.remove_columns(key)
                    datas.append(aa)
                    data_names += file + '\n'
    assert len(datas) > 0, "Invalid file"
    if data_names:
        print("Merge Dataset: \n", data_names)

    data = concatenate_datasets(datas)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if rank == 0:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=16)
            .filter(lambda x: len(x["input_ids"]) < filter_nums)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, num_proc=16)
            .filter(lambda x: len(x["input_ids"]) < filter_nums)
        )
    else:
        train_data = data.shuffle().map(generate_and_tokenize_prompt, num_proc=16)\
            .filter(lambda x: len(x["input_ids"]) < filter_nums)
        val_data = None

    tlist = np.array([len(x) for x in train_data['input_ids']])

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n\nTotal item: ", len(tlist))
        print("cut-off: ", cutoff_len)
        print("max-len token: ", np.max(tlist))
        print("min-len token: ", np.min(tlist))
        for target in [512, 1024, 2048, 4096]:
            count = np.sum(tlist > target)
            print(f"more than {target}-len: {count}")
        print(" \n")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1000 if val_set_size > 0 else None,
            save_steps=20000000,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            run_name="testhahha",
            lr_scheduler_type='cosine',
            deepspeed=deepspeed,
            local_rank=rank,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    if rank == 0:
        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )

if __name__ == "__main__":
    fire.Fire(train)

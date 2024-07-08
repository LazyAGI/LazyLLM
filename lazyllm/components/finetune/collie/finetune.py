import os
import time
import json
import argparse
import datetime
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from prompter import Prompter

from collie import Trainer, EvaluatorForPerplexity, CollieConfig, PPLMetric, CollieDatasetForTraining, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, StepTimeMonitor, InternLMForCausalLM, Callback, \
    ChatGLM2ForCausalLM, ChatGLMForCausalLM, LlamaForCausalLM, InternLM2ForCausalLM

if "SLURM_JOB_ID" in os.environ:
    world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    print("comm: ", world_size, rank, local_rank)
else:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))

os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(rank)
os.environ['LOCAL_RANK'] = str(local_rank)

def proccess(data_path, prompter):
    processed_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            output = item.pop('output', '')
            processed_item = {"text": prompter.generate_prompt(item, label=output)}
            processed_data.append(processed_item)
    return processed_data


class LogCallback(Callback):
    def __init__(self, rank):
        self.iter_time_s = 0
        self.total_time = 0
        self.rank = rank

    def on_train_epoch_begin(self, trainer):
        if self.rank == 0:
            print(f"Epoch: {trainer.epoch_idx:04d}", flush=True)

    def on_train_batch_begin(self, trainer, batch):
        self.iter_time_s = time.time()

    def on_train_batch_end(self, trainer, loss):
        if self.rank == 0:
            v = time.time() - self.iter_time_s
            self.total_time += v
            avg_time_per_iter = self.total_time / (trainer.batch_idx + 1 + trainer.epoch_idx * trainer.steps_per_epoch)
            remaining_iters = trainer.steps_per_epoch - (trainer.batch_idx + 1) + \
                (trainer.config.train_epochs - trainer.epoch_idx - 1) * trainer.steps_per_epoch
            remaining_time = remaining_iters * avg_time_per_iter

            remaining_time_str = str(datetime.timedelta(seconds=round(remaining_time)))
            total_time_str = str(datetime.timedelta(seconds=round(self.total_time)))

            output = subprocess.check_output(['nvidia-smi',
                                              '--query-gpu=memory.used',
                                              '--format=csv,nounits,noheader'])
            memory_usage = {}
            memories = output.decode('utf-8').strip().split('\n')
            for i, mem in enumerate(memories):
                memory_usage[i] = round(float(mem) / 1024, 1)

            total_batch = trainer.engine.train_batch_size()
            item_per_second = total_batch / v

            print((f"Epoch:{trainer.epoch_idx+1:02d} "
                   f"Iter/Total:{trainer.batch_idx+1:04d}/{trainer.steps_per_epoch} "
                   f"Time:{v:.3f} "
                   f"Items/s:{item_per_second:.3f} "
                   f"Cost:{total_time_str}/End:{remaining_time_str} "
                   f"Loss: {loss:.5f} "
                   f"Mem(G): {memory_usage}"
                   ),
                  flush=True)

def main(): # noqa C901
    if rank == 0:
        print(
            f"Training SFT model with params:\n"
            f"base_model: {args.base_model}\n"
            f"data_path: {args.data_path}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"dp_size: {args.dp_size}\n"
            f"pp_size: {args.pp_size}\n"
            f"tp_size: {args.tp_size}\n"
            f"num_epochs: {args.num_epochs}\n"
            f"micro_batch_size: {args.micro_batch_size}\n"
            f"eval_batch_size: {args.eval_batch_size}\n"
            f"batch_size: {args.batch_size}\n"
            f"eval_per_n_steps: {args.eval_per_n_steps}\n"
            f"output_dir: {args.output_dir}\n"
            f"log_tag: {args.log_tag}\n"
            f"zero_stage: {args.zero_stage}\n"
            f"ds_fp16: {args.ds_fp16}\n"
            f"lora_r: {args.lora_r}\n"
            f"lora_alpha: {args.lora_alpha}\n"
            f"lora_dropout: {args.lora_dropout}\n"
            f"lora_target_modules: {args.lora_target_modules}\n"
            f"modules_to_save: {args.modules_to_save}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"model_type: {args.model_type}\n"
        )

    if (world_size != args.dp_size * args.pp_size * args.tp_size):
        dp_size = world_size // (args.pp_size * args.tp_size)
        if rank == 0:
            print(
                "The world size is not equal to the product of the parallel sizes set."
                f"{world_size} != {args.dp_size} * {args.pp_size} * {args.tp_size}.\n"
                f"Set dp_size to {dp_size}."
            )
        args.dp_size = dp_size
    gradient_accumulation_steps = args.batch_size // (args.dp_size * args.micro_batch_size)

    config = CollieConfig.from_pretrained(args.base_model, trust_remote_code=True)
    config.dp_size = args.dp_size
    config.pp_size = args.pp_size
    config.tp_size = args.tp_size
    config.train_epochs = args.num_epochs
    config.train_micro_batch_size = args.micro_batch_size
    config.eval_batch_size = args.eval_batch_size
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.eval_per_n_steps = args.eval_per_n_steps

    config.ds_config = {
        "fp16": {
            "enabled": args.ds_fp16
        },
        "monitor_config": {
            "enabled": True,
            "tag": args.log_tag,
            "tensorboard": {
                "enabled": True,
                "output_path": args.output_dir,
            },
        },
        "zero_optimization": {"stage": args.zero_stage},
    }

    def str2list(string):
        items = string.strip().strip("[]").split(",")
        return [x.strip() for x in items]

    if args.lora_target_modules:
        lora_target_modules = str2list(args.lora_target_modules)
    else:
        lora_target_modules = args.lora_target_modules
    if args.modules_to_save:
        modules_to_save = str2list(args.modules_to_save)
    else:
        modules_to_save = args.modules_to_save

    config.peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    config.seed = 1024
    if args.model_type:
        model_type = args.model_type.lower()
    else:
        model_type = config.model_type.lower()
    if model_type == 'internlm':
        model_cls = InternLMForCausalLM
    if model_type == 'internlm2':
        model_cls = InternLM2ForCausalLM
    elif model_type == 'chatglm1':
        model_cls = ChatGLMForCausalLM
    elif model_type == 'chatglm' or model_type == 'chatglm2':
        model_cls = ChatGLM2ForCausalLM
    else:
        model_cls = LlamaForCausalLM

    model = model_cls.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    prompter = Prompter.from_template(args.prompt_template_name, show=args.show_prompt)
    train_data = proccess(args.data_path, prompter)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    train_dataset = CollieDatasetForTraining(train_data, tokenizer)
    eval_dataset = train_dataset[-32:]

    if rank == 0:
        print("Show case 0:\n", train_data[0], '\n', train_dataset[0])

    def get_token_length(x):
        return len(x['input_ids'])

    tlist = np.array(list(map(get_token_length, train_dataset)))

    if rank == 0:
        print("\n\nTotal item: ", len(tlist))
        print("Total token: ", np.sum(tlist))
        print("max-len token: ", np.max(tlist))
        print("min-len token: ", np.min(tlist))

        targets = [512, 1024, 2048, 4096]
        counts = [(target, np.sum(tlist > target)) for target in targets]
        for target, count in counts:
            print(f"more than {target}-len: {count}")
        print(" \n")

    monitors = [
        StepTimeMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LossMonitor(config),
        EvalMonitor(config)
    ]

    evaluator_ppl = EvaluatorForPerplexity(
        model=model,
        config=config,
        dataset=eval_dataset,
        monitors=[
            EvalMonitor(config)
        ],
        metrics={
            "ppl": PPLMetric()
        },
    )

    callbacks = [LogCallback(rank)]

    trainer = Trainer(
        model=model,
        lr_scheduler=lr_scheduler,
        config=config,
        optimizer=optimizer,
        train_dataset=train_dataset,
        monitors=monitors,
        evaluators=[evaluator_ppl],
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_peft(path=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        help="Base model for SFT.")
    parser.add_argument("--output_dir", type=str,
                        help="Output path for save lora ckpt.")
    parser.add_argument("--data_path", type=str,
                        help="Data path for SFT.")
    parser.add_argument("--dp_size", type=int, default=1,
                        help="DP size of Model.")
    parser.add_argument("--pp_size", type=int, default=1,
                        help="PP size of Model.")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="TP size of Model.")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Train epochs.")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                        help="Train micro batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Eval batch size.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for SFT.")
    parser.add_argument("--eval_per_n_steps", type=int, default=100,
                        help="Eval per n steps")
    parser.add_argument("--log_tag", type=str, default="tb_log",
                        help="Log file name.")
    parser.add_argument("--zero_stage", type=int, default=1,
                        help="Stage for ZeRO.")
    parser.add_argument("--ds_fp16", type=bool, default=True,
                        help="Deepspeed config for fp16 enabled, default=True.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for SFT.")
    parser.add_argument("--prompt_template_name", type=str, default='alpaca',
                        help="Prompt template name, default=alpaca.")
    parser.add_argument("--show_prompt", type=bool, default=False,
                        help="show prompt or not, default=False.")
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--modules_to_save", type=str, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--model_type", type=str, default=None,
                        help=("Base model type for SFT. Support: "
                              "InternLM1, LLaMa1-2, ChatGLM1-2"))

    args = parser.parse_args()

    main()

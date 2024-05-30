import os
import argparse

import torch
from transformers import AutoModelForCausalLM

from peft import PeftModel


def main(args):
    if "SLURM_JOB_ID" in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        os.environ['RANK'] = str(proc_id)

    if int(os.environ.get("RANK", 0)) == 0:
        merged_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                args.base,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # device_map="auto"
            ),
            args.adapter,
            torch_dtype=torch.float16,
        ).merge_and_unload()
        merged_model.save_pretrained(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", "-m", type=str, default="origin")
    parser.add_argument("--adapter", "-a", type=str, default="Stage2")
    parser.add_argument("--save_path", "-d", type=str, default="Merge_Stage2")
    args = parser.parse_args()
    main(args)

# python -u merge_weights.py -m internlm/internlm-7b -a ./lora/default/last -d ./lora/default/merged

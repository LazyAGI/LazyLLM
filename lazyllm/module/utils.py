import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import lazyllm
from lazyllm.thirdparty import datasets
from ..components.utils.file_operate import delete_old_files

@dataclass
class TrainConfig:
    finetune_model_name: str = 'llm'
    base_model: str = 'llm'
    training_type: str = 'SFT'
    finetuning_type: str = 'LoRA'
    data_path: str = 'path/to/dataset'
    num_gpus: int = 1
    val_size: float = 0.1
    num_epochs: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = 'cosine'
    batch_size: int = 32
    cutoff_len: int = 1024
    lora_r: int = 8
    lora_alpha: int = 32
    lora_rate: float = 0.1

def update_config(input_dict: dict, default_data: type) -> dict:
    config = TrainConfig()
    config_dict = asdict(config)
    assert all([key in config_dict for key in input_dict.keys()]), \
        f"The {input_dict.keys()} must be the subset of {config_dict.keys()}."
    config_dict.update(input_dict)
    return config_dict

INPUT_SPLIT = " ### input "

def uniform_sft_dataset(dataset_path: str, target: str = 'alpaca') -> str:
    '''
    {origin_format}.{suffix} -> {target_format}, supported all 8 cases:
    1. openai.json   -> alpaca: Conversion: openai2alpaca: json
    2. openai.jsonl  -> alpaca: Conversion: openai2alpaca: json
    3. alpaca.json   -> alpaca: Keep: json
    4. alpaca.jsonl  -> alpaca: Restore: jsonl -> json
    5. openai.json   -> openai: Restore: json -> jsonl
    6. openai.jsonl  -> openai: Keep: jsonl
    7. alpaca.json   -> openai: Conversion: alpaca2openai: jsonl
    8. alpaca.jsonl  -> openai: Conversion: alpaca2openai: jsonl
    Note: target-suffix does match:{'openai': 'jsonl'; 'alpaca': 'json'}
    '''
    assert os.path.exists(dataset_path), f"Path: {dataset_path} does not exist!"

    data = datasets.load_dataset('json', data_files=dataset_path)
    file_name = os.path.basename(dataset_path)
    base_name, suffix = file_name.split('.')
    assert suffix in ['json', 'jsonl']
    target = target.strip().lower()
    save_suffix = 'json'

    # Get the format('alpaca' or 'openai') of the original dataset
    origin_format = 'alpaca'
    if "messages" in data["train"][0]:
        origin_format = 'openai'

    # Verify that the dataset format is consistent with the target format
    if origin_format == target:
        if target == 'alpaca':
            if suffix == 'json':
                return dataset_path
            else:
                save_data = alpaca_filter_null(data)
        else:
            if suffix == 'jsonl':
                return dataset_path
            else:
                save_suffix = 'jsonl'
                save_data = data['train'].to_list()
    else:
        # The format is inconsistent, conversion is required
        if target == 'alpaca':
            save_data = openai2alpaca(data)
        elif target == 'openai':
            save_data = alpaca2openai(data)
            save_suffix = 'jsonl'
        else:
            raise ValueError(f"Not supported type: {target}")

    return save_dataset(save_data, save_suffix, base_name + f'_{suffix}')

def save_json(data: list, output_json_path: str) -> None:
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def save_jsonl(data: list, output_json_path: str) -> None:
    with open(output_json_path, mode='w', encoding='utf-8') as json_file:
        for row in data:
            json_file.write(json.dumps(row, ensure_ascii=False) + '\n')

def save_dataset(save_data: list, save_suffix='json', base_name='train_data') -> str:
    directory = os.path.join(lazyllm.config['temp_dir'], 'dataset')
    if not os.path.exists(directory):
        os.makedirs(directory)
    delete_old_files(directory)
    time_stamp = datetime.now().strftime('%y%m%d%H%M%S%f')[:14]
    output_json_path = os.path.join(directory, f'{base_name}_{time_stamp}.{save_suffix}')
    if save_suffix == 'json':
        save_json(save_data, output_json_path)
    else:
        save_jsonl(save_data, output_json_path)
    return output_json_path

def alpaca_filter_null(data) -> list:
    res = []
    for item in data["train"]:
        alpaca_item = dict()
        for key in item.keys():
            if item[key]:
                alpaca_item[key] = item[key]
        res.append(alpaca_item)
    return res

def alpaca2openai(data) -> list:
    res = []
    for item in data["train"]:
        openai_item = {"messages": []}
        inp = item.get("input", "")
        system = item.get("system", "")  # Maybe get None
        historys = item.get("history", [])
        if system:
            openai_item["messages"].append({"role": "system", "content": system})
        openai_item["messages"].extend([
            {"role": "user", "content": item["instruction"] + (INPUT_SPLIT + inp if inp else "")},
            {"role": "assistant", "content": item["output"]}
        ])
        if historys:
            for history in historys:
                openai_item["messages"].append({"role": "user", "content": history[0]})
                openai_item["messages"].append({"role": "assistant", "content": history[1]})

        res.append(openai_item)

    return res

def openai2alpaca(data) -> list:
    res = []
    for line in data["train"]:
        chat = line["messages"]
        system = ''
        instructions = []
        outputs = []
        for item in chat:
            if item["role"] == "system" and not system:
                system = item["content"]
            if item["role"] == "user":
                instructions.append(item["content"])
            if item["role"] == "assistant":
                outputs.append(item["content"])
        assert len(instructions) == len(outputs)
        history = [[x, y] for x, y in zip(instructions[1:], outputs[1:])]
        instruction_input = instructions[0].split(INPUT_SPLIT)
        instruction = instruction_input[0]
        inp = ''
        if len(instruction_input) >= 2:
            inp = instruction_input[-1]
        output = outputs[0]
        alpaca_item = dict()
        if system:
            alpaca_item["system"] = system
        alpaca_item["instruction"] = instruction
        # fixed llama-factory-bug: must have input
        alpaca_item["input"] = inp
        alpaca_item["output"] = output
        if history:
            alpaca_item["history"] = history
        res.append(alpaca_item)
    return res

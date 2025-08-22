import os
import re
import json
import argparse

import lazyllm
from lazyllm import finetune, deploy, launchers, warp

from modelscope.msdatasets import MsDataset


def load_data(data_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def save_res(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def build_data_path(file_name):
    data_root = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    save_path = os.path.join(data_root, file_name)
    return save_path

def get_dataset(dataset_name):
    train_path = build_data_path('train_set.json')
    eval_path = build_data_path('eval_set.json')
    ds = MsDataset.load(dataset_name, subset_name='main')
    ds = ds.rename_column('question', 'instruction').rename_column('answer', 'output')
    with open(train_path, 'w') as file:
        json.dump(ds['train'].to_list(), file, ensure_ascii=False, indent=4)
    with open(eval_path, 'w') as file:
        json.dump(ds['test'].to_list(), file, ensure_ascii=False, indent=4)
    return train_path, eval_path

def distill_dataset(data_path, model=None, demo=False):
    inputs = load_data(data_path)[:1] if demo else load_data(data_path)
    with warp(_concurrent=1) as wp:
        wp.func = model
    res_list = []
    try_n = 0
    while inputs:
        print(">>>" * 12, f"{try_n + 1} times left: ", len(inputs))
        querys = [item['instruction'] for item in inputs]
        results = wp(querys)
        valid_data, inputs = filter(inputs, results)
        res_list.extend(valid_data)
        try_n += 1
        if try_n == 15:
            break
    res_list = res_list * 120 if demo else res_list
    distilled_train_set_path = build_data_path('distilled_train_data.json')
    save_res(res_list, distilled_train_set_path)
    save_res(inputs, build_data_path('left_data.json'))
    return distilled_train_set_path

def filter(inputs, results):
    valid = []
    retry = []
    for i, item in enumerate(inputs):
        true_v = item['output'].split('\n#### ')[-1].strip()
        if f'\\boxed{{{true_v}}}' in results[i] and '</think>' in results[i]:
            valid.append({'instruction': item['instruction'], 'output': results[i], 'input': ''})
        else:
            retry.append(item)
    return valid, retry

def extract_boxed_content(text):
    pattern = r'boxed{((?:[^{}]*|{.*?})*)}'
    contents = re.findall(pattern, text)
    return contents

def caculate_score(eval_set, infer_set):
    assert len(eval_set) == len(infer_set)
    score = 0
    for index, eval_item in enumerate(eval_set):
        output = infer_set[index]
        if 'boxed{' in output:
            res = extract_boxed_content(output)
            res = list(set(res))
            res = res[0] if len(res) == 1 else res
            if type(res) is list:
                continue
            true_v = eval_item['output'].split('\n#### ')[-1].strip()
            if true_v == res.strip():
                score += 1
    return f'{score}/{len(eval_set)}, {round(score / len(eval_set), 4) * 100}%'

def main(techer_name, student_name, dataset_name, demo=False, sft_data_path=None):
    # Launcher Teacher
    teacher_model = lazyllm.OnlineChatModule(techer_name)

    # Load and Distill Dataset
    train_set_path, eval_set_path = get_dataset(dataset_name)
    eval_set = load_data(eval_set_path)
    if not sft_data_path:
        sft_data_path = distill_dataset(train_set_path, teacher_model, demo)

    # Train and Infer
    infer_data = [item['instruction'] for item in eval_set]
    student_model = lazyllm.TrainableModule(student_name)\
        .mode('finetune')\
        .trainset(sft_data_path)\
        .finetune_method((finetune.llamafactory, {
            'learning_rate': 1e-4,
            'cutoff_len': 5120,
            'max_samples': 20000,
            'val_size': 0.01,
            'per_device_train_batch_size': 2,
            'num_train_epochs': 2.0,
            'launcher': launchers.sco(nnode=1, nproc=8, ngpus=8)
        }))\
        .prompt(dict(system='You are a helpful assistant.', drop_builtin_system=True))\
        .deploy_method(deploy.Vllm)
    student_model._prompt._soa = '<|im_start|>assistant\n\n<think>'
    student_model.evalset(infer_data)
    student_model.update()

    # Score
    score = caculate_score(eval_set, student_model.eval_result)
    print("All Done. Score is: ", score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distill the model training script with given parameters.")
    parser.add_argument('--teacher_model_name', type=str, default='DeepSeek-R1', help='Name of the teacher model')
    parser.add_argument('--student_model_name', type=str, default='internlm2-chat-7b', help='Name of the student model')
    parser.add_argument('--dataset_name', type=str, default='modelscope/gsm8k', help='Name of the dataset')
    parser.add_argument('--demo', type=bool, default=True, help='Demo mode flag')
    parser.add_argument('--sft_data_path', type=str, default=None, help='Path to the SFT data')

    args = parser.parse_args()

    # Extracting arguments
    teacher_model_name = args.teacher_model_name
    student_model_name = args.student_model_name
    dataset_name = args.dataset_name
    demo = args.demo
    sft_data_path = args.sft_data_path

    # Calling the main function
    main(teacher_model_name, student_model_name, dataset_name, demo, sft_data_path)

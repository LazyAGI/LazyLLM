import os
import csv
import json
import pandas as pd
from typing import List
from datetime import datetime
from datasets import load_dataset

import lazyllm
from lazyllm.module.utils import openai2alpaca
from lazyllm.components.utils.file_operate import delete_old_files

# origin_key: target_key:
default_mapping = {'instruction': 'instruction', 'input': 'input', 'output': 'output'}

def csv2alpaca(dataset_path: str, header_mapping=None, target_path: str = None) -> str:
    """
    Convert a CSV file to a JSON file with custom header mapping.

    :param dataset_path: path of the CSV file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    :param target_path: The path of the folder where the converted files are stored.
        The default is None, and it will be stored in the working path + `.temp/dataset`.
    """
    save_dir = _build_target_dir(target_path)

    mapping = header_mapping if header_mapping else default_mapping
    data = []
    with open(dataset_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            renamed_row = {mapping.get(k, k): v for k, v in row.items() if k in mapping}
            data.append(renamed_row)
    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')

    res_path = _save_dataset(data, save_dir, base_name)
    return res_path

def parquet2alpaca(dataset_path: str, header_mapping=None, target_path: str = None) -> str:
    """
    Convert a Parquet file to a JSON file with custom header mapping.

    :param dataset_path: path of the Parquet file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    :param target_path: The path of the folder where the converted files are stored.
        The default is None, and it will be stored in the working path + `.temp/dataset`.
    """
    save_dir = _build_target_dir(target_path)

    mapping = header_mapping if header_mapping else default_mapping
    df = pd.read_parquet(dataset_path)

    df = df.rename(columns=mapping)
    df = df[[col for col in mapping.values()]]
    data = df.to_dict(orient='records')

    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')

    res_path = _save_dataset(data, save_dir, base_name)
    return res_path

def json2alpaca(dataset_path: str, header_mapping=None, target_path: str = None) -> str:
    """
    Convert a JSON file to a JSON file with custom header mapping.

    :param dataset_path: path of the JSON file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    :param target_path: The path of the folder where the converted files are stored.
        The default is None, and it will be stored in the working path + `.temp/dataset`.
    """
    save_dir = _build_target_dir(target_path)

    mapping = header_mapping if header_mapping else default_mapping
    dataset = load_dataset('json', data_files=dataset_path)

    data_list = []
    for row in dataset['train']:
        renamed_row = {mapping.get(k, k): v for k, v in row.items() if k in mapping}
        data_list.append(renamed_row)

    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')

    res_path = _save_dataset(data_list, save_dir, base_name)
    return res_path

def merge2alpaca(dataset_paths: List[str], target_path: str = None) -> str:
    """
    Merge multiple JSON files into a single JSON file formatted for Alpaca.
    This function reads multiple JSON files(Alpaca or OpenAI format), converts them to Alpaca format.
    The merged file is saved to the specified target directory or to a default temporary directory
    if no target is provided.

    :param dataset_paths: A list of paths to the JSON files to be merged.
    :param target_path: The path of the folder where the merged file will be stored.
        If `None`, the file will be stored in the working path + `.temp/merged`.
    :return: The path to the merged dataset file.

    Raises:
        RuntimeError: If any of the provided file paths do not exist.
    """
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    non_existent_files = [path for path in dataset_paths if not os.path.exists(path)]
    if non_existent_files:
        raise RuntimeError(f"These files does not exist at {non_existent_files}")
    save_dir = _build_target_dir(target_path)

    merge_list = []
    for path in dataset_paths:
        data = load_dataset('json', data_files=path)
        if "messages" in data["train"][0]:
            alpaca_data = openai2alpaca(data)
            merge_list.extend(alpaca_data)
        else:
            merge_list.extend(data['train'].to_list())
    res_path = _save_dataset(merge_list, save_dir, 'merge_dataset')
    return res_path

def _build_target_dir(target_path: str = None) -> str:
    if target_path:
        save_dir = target_path
        if not os.path.exists(save_dir):
            raise RuntimeError(f"The target_path at {save_dir} does not exist.")
    else:
        save_dir = os.path.join(lazyllm.config['temp_dir'], 'dataset')
        if not os.path.exists(save_dir):
            os.system(f'mkdir -p {save_dir}')
        else:
            delete_old_files(save_dir)
    return save_dir

def _save_dataset(data: list, save_dir: str, base_name: str) -> str:
    time_stamp = datetime.now().strftime('%y%m%d%H%M%S%f')[:14]
    output_json_path = os.path.join(save_dir, f'{base_name}_{time_stamp}.json')
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    return output_json_path

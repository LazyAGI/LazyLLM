import os
import csv
import pandas as pd
from datasets import load_dataset

from lazyllm.module.utils import save_dataset

# origin_key: target_key:
default_mapping = {'instruction': 'instruction', 'input': 'input', 'output': 'output'}

def csv2alpaca(dataset_path: str, header_mapping=None) -> str:
    """
    Convert a CSV file to a JSON file with custom header mapping.

    :param dataset_path: path of the CSV file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    """
    mapping = header_mapping if header_mapping else default_mapping
    data = []
    with open(dataset_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            renamed_row = {mapping.get(k, k): v for k, v in row.items() if k in mapping}
            data.append(renamed_row)
    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')
    res_path = save_dataset(data, base_name=base_name)
    return res_path

def parquet2alpaca(dataset_path: str, header_mapping=None) -> str:
    """
    Convert a Parquet file to a JSON file with custom header mapping.

    :param dataset_path: path of the Parquet file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    """
    mapping = header_mapping if header_mapping else default_mapping
    df = pd.read_parquet(dataset_path)

    df = df.rename(columns=mapping)
    df = df[[col for col in mapping.values()]]
    data = df.to_dict(orient='records')

    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')
    res_path = save_dataset(data, base_name=base_name)
    return res_path

def json2alpaca(dataset_path: str, header_mapping=None) -> str:
    """
    Convert a JSON file to a JSON file with custom header mapping.

    :param dataset_path: path of the JSON file to be converted.
    :param header_mapping: A dictionary representing the header mapping. Default is None.
    """
    mapping = header_mapping if header_mapping else default_mapping
    dataset = load_dataset('json', data_files=dataset_path)

    data_list = []
    for row in dataset['train']:
        renamed_row = {mapping.get(k, k): v for k, v in row.items() if k in mapping}
        data_list.append(renamed_row)

    file_name = os.path.basename(dataset_path)
    base_name, _ = file_name.split('.')
    res_path = save_dataset(data_list, base_name=base_name)
    return res_path


if __name__ == '__main__':
    #  test: csv
    res = csv2alpaca('alpaca_gpt4_data_zh.csv')
    print(res)

    #  test: parquet
    res = parquet2alpaca('tatsu-lab-alpaca_head_100.parquet')
    print(res)

    #  test: jsonl
    header_mapping = {'problem': 'instruction', 'solution': 'output'}
    res = json2alpaca('geometry.jsonl', header_mapping)
    print(res)

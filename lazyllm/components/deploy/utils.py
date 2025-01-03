import os
from datetime import datetime
import random
import lazyllm


def make_log_dir(log_path, framework):
    folder = log_path or os.path.join(lazyllm.config['temp_dir'], 'deploy_log', framework)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_log_path(folder_path):
    formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_value = random.randint(1000, 9999)
    return f'{folder_path}/infer_{formatted_date}_{random_value}.log'

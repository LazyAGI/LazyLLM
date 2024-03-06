import os
import bisect
import warnings
import pkg_resources
from pprint import pprint

from lazyllm import launchers, finetune
from .auto.modelspool import models_pool
from .auto.alpacalora_info import alpaca_lora_info
from .auto.collie_info import collie_info
from .base import LazyLLMFinetuneBase


class AutoFinetune(LazyLLMFinetuneBase):
    def __new__(
        cls,
        model_name,
        gpu_count,
        target_path=None,
        base_model=None,
        **kwargs,
    ):
       assigner = Assigner()
       return assigner.assign_finetuner(
            model_name,
            gpu_count,
            target_path,
            base_model,
            **kwargs,
       )
    
class Assigner(object):
    def __init__(self,):
        self.gpu_memory = None
        self.launcher = None

        self.models_name_set = None
        self.model_name = None
        self.model_size = None
        self.model_config = dict()

        self.frame_infor = dict()
        self.frame_config = dict()
        self.frame_ability = dict()

        self.traing_config = dict()
        self.launch_config = dict()

        self.get_default_info()

        # Base traning config
        self.lora_path = None
        self.merge_path = None
        self.base_model_path = None
        self.frame = None

    def get_default_info(self):
        # Get Env info
        if "LAZYLLM_LAUNCHER" in os.environ:
            self.launcher = os.environ["LAZYLLM_LAUNCHER"].lower()
        else:
            warnings.warn("Environment variable 'LAZYLLM_LAUNCHER' "
                          "not found, setting it to 'slurm'")
            self.launcher = "slurm"
        if "LAZYLLM_GPU_MEMORY" in os.environ:
            try:
                self.gpu_memory = int(os.environ["LAZYLLM_GPU_MEMORY"])
            except ValueError:
                warnings.warn("Value of environment variable 'LAZYLLM_GPU_MEMORY'"
                              " is not an integer, setting it to '81'")
                self.gpu_memory = 81
        else:
            warnings.warn("Environment variable 'LAZYLLM_GPU_MEMORY' "
                          "not found, setting it to '81'")
            self.gpu_memory = 81

        # Build models name set
        model_names = set()
        for base_name, info in models_pool.items():
            for key, _ in info.items():
                if key not in ['lora_config']:
                    model_names.add(f"{base_name}-{key}")
        self.models_name_set = model_names

        # Get Frame info
        self.frame_infor['alpacalora'] = alpaca_lora_info
        self.frame_infor['collie'] = collie_info

    def get_frame_list(self, model_name, gpu_count):
        # Check model in models_set
        model_res = model_name.strip().lower()
        assert model_res in self.models_name_set, (
                f"Not support model: {model_name}\n"
                f"Support models are: {self.models_name_set}" 
                )
        self.model_name = model_res

        # Get info from modelpool
        base_name, suffix = self.model_name.split('-', 1)
        model_group = models_pool[base_name]
        self.model_config.update(model_group['lora_config'])
        self.model_config.update(model_group[suffix]['addition_parameters'])
        self.model_size = model_group[suffix]['gpu_memory']
        frame_propose = model_group[suffix]['frame_propose']

        # Check the GPU memory consumption under Zero2 during training
        #TODO(sunxiaoye): Carefully calculate by model and strategy.
        activation = self.model_size*0.2
        lora_model_size = self.model_size*0.45
        gradients = lora_model_size/gpu_count
        optimizers = lora_model_size*6/gpu_count

        res = self.model_size + gradients + optimizers + activation
        assert res <= self.gpu_memory, (
            f"One GPU memory usage maybe {res} G, "
            f"which less than a gpu memory {self.gpu_memory} G."
            )
        #TODO(sunxiaoye): Log sys
        print(f"One GPU memory usage maybe {res} G.")

        # Get frame list
        numbers = [1, 2, 4, 8, 16]
        num_index = bisect.bisect_left(numbers, gpu_count)
        return frame_propose[numbers[num_index]]

    def check_requirements(self, requirements):
        packages = [line.strip() for line in requirements.split('\n') if line.strip()]

        not_installed = []
        for package in packages:
            parts = package.split('==') if '==' in package else package.split('>=') if '>=' in package else [package]
            try:
                installed = pkg_resources.get_distribution(parts[0])
                if len(parts) > 1:
                    if parts[1] not in installed.version:
                        not_installed.append(f"{package} (Installed: {installed.version}, Required: {parts[1]})")
            except pkg_resources.DistributionNotFound:
                not_installed.append(f"Required: {package}")
        return not_installed
    
    def get_training_frame(self, frame_list):
        for frame in frame_list:
            assert frame in self.frame_infor, f"Framework {frame} not in {self.frame_infor.keys()}"
            not_installed = self.check_requirements(self.frame_infor[frame]['requrements'])
            if not_installed:
                #TODO(sunxiaoye): Log sys
                print(f"Not support {frame}, lack folllowing dependencies:\n\t- "
                        +'\n\t- '.join(not_installed))
            else:
                self.frame_config.update(self.frame_infor[frame]['train_params'])
                self.frame_ability.update(self.frame_infor[frame]['ability'])
                return frame

    def set_training_config(self, gpu_count, target_path, base_model):
        # Load frame defatult config
        self.traing_config = self.frame_config.copy()

        # Update model specific config
        self.traing_config.update(self.model_config)
        self.traing_config['model_name'] = self.model_name

        # Set dataset config
        gradient_accumulation_steps = self.traing_config["batch_size"] / \
            self.traing_config["micro_batch_size"] / gpu_count
        if gradient_accumulation_steps < 1:
            self.traing_config["batch_size"] = \
                self.traing_config["micro_batch_size"] * gpu_count
            
        # Set launch config
        launch_config = dict()
        if self.launcher == "slurm":
            launch_config = {
                'partition': os.environ.get('LAZYLLM_PARTITION', 'pat_rd'),
                'nnode': -(-gpu_count // 8),
                'nproc': gpu_count,
                'ngpus': 8,
            }
            if launch_config["nnode"] > 1:
                launch_config["ngpus"] = 8
            else:
                launch_config["ngpus"] = gpu_count
        self.launch_config.update(launch_config)

        # Build save path
        _, self.lora_path, self.merge_path = self.create_directories(target_path, self.model_name)

        # Get base_model path
        if base_model and os.path.exists(base_model):
            self.base_model_path = base_model
        else:
            base_path = os.getenv("LAZYLLM_MODELS_HOME", None)
            assert base_model, "LAZYLLM_MODELS_HOME environment variable is not set."
            base_model_path = os.path.join(base_path, self.model_name)
            assert os.path.exists(base_model_path), f"Cannot find base model path: {base_model_path}"
            self.base_model_path = base_model_path

    def assign_finetuner(self,
                         model_name,
                         gpu_count,
                         target_path,
                         base_model,
                         **kwargs,
                         ):
        frame_list = self.get_frame_list(model_name, gpu_count)
        self.frame = self.get_training_frame(frame_list)
        self.set_training_config(gpu_count, target_path, base_model)
        self.show_training_config()

        finetune_cls = getattr(finetune, self.frame)
        launcher_cls = getattr(launchers, self.launcher)
        return finetune_cls(
            base_model=self.base_model_path,
            target_path=self.lora_path,
            merge_path=self.merge_path,
            **self.traing_config,
            launcher=launcher_cls(**self.launch_config),
        )

    def show_training_config(self):
        #TODO(sunxiaoye): Log sys
        print('\n'+'=='*20)
        print("AutoFinetune Config:\n"
              f"\t Base Model Path: {self.base_model_path}\n"
              f"\t Lora Path: {self.lora_path}\n"
              f"\t Merge Path: {self.merge_path}\n"
              )
        pprint(self.traing_config)
        pprint(self.launch_config)
        print('=='*20+'\n')

    @classmethod
    def create_directories(self, target_path, model_name):
        if target_path is None:
            target_path = os.path.join(os.getcwd(), model_name)
        lora_path, merge_path = os.path.join(target_path, "lora"), os.path.join(target_path, "merge")
        os.system(f'mkdir -p {target_path} {lora_path} {merge_path}')
        return target_path, lora_path, merge_path

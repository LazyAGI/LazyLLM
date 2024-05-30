from core import AutoConfig
from core.configuration import TrainingConfiguration, DeployConfiguration
import unittest


class TestTrainingMethods(unittest.TestCase):
    def test_load_and_query(self):
        input = dict(
            gpu_type="A100",
            gpu_num=5,
            model_name="LLAMA_7B",
            ctx_len=512,
            batch_size=32,
            lora_r=8,
        )
        output = [
            TrainingConfiguration(
                framework="EASYLLM",
                tp=4,
                zero=True,
                gradient_step=1,
                sp=1,
                ddp=1,
                micro_batch_size=32,
                tgs=100,
            )
        ]
        r = AutoConfig(finetune_file="sample_finetune.csv", deploy_file="sample_deploy.csv").query_finetune(**input)
        self.assertEqual(r, output)


class TestDeployMethods(unittest.TestCase):
    def test_load_and_query(self):
        input = dict(
            gpu_type="A100",
            gpu_num=4,
            model_name="LLAMA_7B",
            max_token_num=1024,
        )
        output = [
            DeployConfiguration(
                framework='VLLM',
                tp=4,
                tgs=100,
            )
        ]
        r = AutoConfig(finetune_file="sample_finetune.csv", deploy_file="sample_deploy.csv").query_deploy(**input)
        self.assertEqual(r, output)

if __name__ == "__main__":
    unittest.main()

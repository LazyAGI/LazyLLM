from configuration import (
    HardwareConfiguration,
    TrainingConfiguration,
    DeployConfiguration
)
from autoconfigurer import (
    AutoFinetuneConfigurer,
    AutoDeployConfigurer
)
import unittest


class TestTrainingMethods(unittest.TestCase):
    def test_load_and_query(self):
        input = HardwareConfiguration(
            gpu_type="A100",
            gpu_num=5,
            model_name="LLAMA_7B",
            ctx_len=512,
            batch_size=64,
            trainable_params=5,
        )
        output = [
            TrainingConfiguration(
                framework="EASYLLM",
                tp=4,
                pp=1,
                zero=True,
                gradient_step=1,
                lora_r=2,
                sp=1,
                ddp=1,
                micro_batch_size=16,
                memory_usage_gb=40,
                tgs=100,
                additional_arguments="''",
            )
        ]
        db = AutoFinetuneConfigurer("./sample_finetune.csv")
        get = db.query(input)
        self.assertEqual(get, output)


class TestDeployMethods(unittest.TestCase):
     def test_load_and_query(self):
        input = HardwareConfiguration(
            gpu_type="A100",
            gpu_num=4,
            model_name="LLAMA_7B",
            ctx_len=512,
            batch_size=64,
            trainable_params=0,
        )
        output = [
            DeployConfiguration(
                framework='VLLM',
                tp=4,
                max_token_num=1024,
                additional_arguments="''",
                tgs=100,
           )
        ]
        db = AutoDeployConfigurer("./sample_deploy.csv")
        get = db.query(input)
        self.assertEqual(get, output)

if __name__ == "__main__":
    unittest.main()

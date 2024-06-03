# flake8: noqa
from collections.abc import Callable, Generator, Iterator
import csv
import itertools
from core import protocol as proto
import typing
import unittest
from typing import List, Dict, Union

T = typing.TypeVar("T")

def values(rule: proto.Rule[T], default: Union[T, None] = None) -> List[T]:
    if rule.options is None:
        return [default or rule.value_type()]
    return rule.options

def overwrite(input: Dict[str, List[typing.Any]], **kwargs: List[typing.Any]) -> Dict[str, List[typing.Any]]:
    for key, value in kwargs.items():
        assert key in input
        assert set(value) <= set(input[key])
    input.update(**kwargs)
    return input

def combine(options: Dict[str, List[typing.Any]], **constraints: Callable[[Dict[str, typing.Any]], bool])\
        -> Generator[Iterator[typing.Any], typing.Any, None]:
    limit = len(options)
    input = [(key, value) for key, value in options.items()]
    output: Dict[str, typing.Any] = {}

    def f(index: int) -> Generator[Iterator[typing.Any], typing.Any, None]:
        if index == limit:
            yield output.values().__iter__()
        else:
            name, items = input[index]
            for item in items:
                output[name] = item
                if constraints.get(name, lambda _: True)(output):
                    yield from f(index + 1)
                output[name] = None

    yield from f(0)


def generate(filename: str, head: List[typing.Any], body: Iterator[Iterator[typing.Any]]):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        count = 0
        for value in body:
            writer.writerow(value)
            count += 1
        return count


class TestGenerateEmptyCSV(unittest.TestCase):
    def test_generate_csv(self): # noqa C901
        rules = proto.TRAINING_RULE_SET
        items = {rule.name: values(rule) for rule in rules}
        items = overwrite(
            items,
            GPU_NUM=[1, 2, 4, 8, 16, 24, 32],
            MODEL_NAME=["LLAMA2_7B", "LLAMA2_13B", "LLAMA2_70B"],
            CTX_LEN=[128, 256, 512, 1024, 2048, 4096],
            BATCH_SIZE=[16],
            TRAINABLE_PARAMS=[0],
            ZERO=[True],
            LORA_R=[0, 8, 16, 32])

        def to_parameter_size(model_name: str) -> int:
            return {
                "LLAMA_7B": 7,
                "LLAMA_13B": 13,
                "LLAMA_20B": 20,
                "LLAMA_65B": 65,
                "LLAMA2_7B": 7,
                "LLAMA2_13B": 13,
                "LLAMA2_70B": 70,
            }[model_name]

        def to_gpu_memory(x: str) -> int:
            return {"A100": 80, "A800": 80}[x]

        def is_inference_framework(x: str) -> bool:
            return x in ["LIGHTLLM", "VLLM"]

        def model_name_constraint(_: Dict[str, typing.Any]) -> bool:
            mapping = {
                "LLAMA_7B": 80,
                "LLAMA_13B": 80 * 2,
                "LLAMA_20B": 80 * 2,
                "LLAMA_65B": 80 * 4,
                "LLAMA2_7B": 80,
                "LLAMA2_13B": 80 * 2,
                "LLAMA2_70B": 80 * 4,
            }
            r = mapping[_["MODEL_NAME"]]
            m = to_gpu_memory(_["GPU_TYPE"])
            options = proto.GPU_NUM.options
            assert options is not None and proto.GPU_NUM.indexed
            le = min(i for i, x in enumerate(options) if r <= x * m)
            r = options.index(_["GPU_NUM"])
            return le <= r <= le + 2

        def tp_constraint(_: Dict[str, typing.Any]) -> bool:
            if _["TP"] > _["GPU_NUM"] or to_parameter_size(_["MODEL_NAME"]) > 30 * _["TP"]:
                return False
            if _["FRAMEWORK"] in ["ALPACA"]:
                return _["TP"] == min(_["GPU_NUM"], 8)
            return True

        def pp_constraint(_: Dict[str, typing.Any]) -> bool:
            if _["TP"] * _["PP"] > _["GPU_NUM"]:
                return False
            if _["FRAMEWORK"] in ["ALPACA"]:
                return _["PP"] == 1
            return True

        def gradient_step_constraint(_: Dict[str, typing.Any]) -> bool:
            options = proto.GRADIENT_STEP.options
            gradient_size = {
                "LLAMA_7B": 4,
                "LLAMA_13B": 4,
                "LLAMA_20B": 4,
                "LLAMA_65B": 4,
                "LLAMA2_7B": 4,
                "LLAMA2_13B": 4,
                "LLAMA2_70B": 4,
            }
            minimium_activation_size = {
                "LLAMA_7B": 3,
                "LLAMA_13B": 5,
                "LLAMA_20B": 5,
                "LLAMA_65B": 6,
                "LLAMA2_7B": 3,
                "LLAMA2_13B": 5,
                "LLAMA2_70B": 6,
            }

            assert options is not None
            for i, gradient_step in enumerate(options):
                require = to_parameter_size(_["MODEL_NAME"]) * 2 / _["TP"] + \
                    gradient_size[_["MODEL_NAME"]] * _["GRADIENT_STEP"] + \
                    minimium_activation_size[_["MODEL_NAME"]] * _["CTX_LEN"] * \
                    _["BATCH_SIZE"] * _["TP"] * _["PP"] / gradient_step / 128.0
                if require < to_gpu_memory(_["GPU_TYPE"]) * _["GPU_NUM"]:
                    return _["GRADIENT_STEP"] in options[i: min(i + 1, len(options))]
            return False

        def lora_r_constraint(_: Dict[str, typing.Any]) -> bool:
            if is_inference_framework(_["FRAMEWORK"]):
                return _["LORA_R"] == 0
            return _["LORA_R"] > 0

        body = combine(items,
                       MODEL_NAME=model_name_constraint,
                       TP=tp_constraint,
                       PP=pp_constraint,
                       GRADIENT_STEP=gradient_step_constraint,
                       LORA_R=lora_r_constraint)

        head = [rule.name for rule in rules]
        size = generate('sample_finetune_empty.csv', head, itertools.islice(body, 1000000))
        print(f"GENERATE {size} cases")


if __name__ == '__main__':
    unittest.main()

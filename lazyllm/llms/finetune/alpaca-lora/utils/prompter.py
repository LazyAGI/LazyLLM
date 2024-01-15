"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "_template_name")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        self._template_name = template_name

        folder_path = osp.dirname(osp.abspath(__file__))
        file_name = osp.join(folder_path, "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
        background: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if background:
            res = self.template["prompt_input_background"].format(
                instruction=instruction, input=input, background=background
            )
        elif input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if self._template_name == 'puyu':
            outputs = output.split(self.template["response_split"])
            if '命令已收到，请告诉我具体要做到事情' in outputs[1]:
                output = outputs[2].strip()
            else:
                output = outputs[1].strip()
            return output.split('ി')[0]
        else:
            return output.split(self.template["response_split"])[1].strip()

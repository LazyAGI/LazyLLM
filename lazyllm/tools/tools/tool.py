from lazyllm.module import ModuleBase
from lazyllm.flow import pipeline
from lazyllm.common import compile_code

class Tool(ModuleBase):
    def __init__(self, module: ModuleBase, post_process_code: str):
        super().__init__()
        if post_process_code:
            with pipeline() as ppl:
                ppl.module = module
                ppl.post_processor = compile_code(post_process_code)
            self.executor = ppl
        else:
            self.executor = module

    def forward(self, *args, **kwargs):
        return self.executor(*args, **kwargs)

from typing import Union, Optional
from ...module import ModuleBase, TrainableModule


class QustionRewrite(ModuleBase):
    def __init__(
        self,
        base_model: Union[str, TrainableModule],
        rewrite_prompt: str = "",
        formatter: str = "list",
    ):
        super().__init__()
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(rewrite_prompt)
        else:
            self._m = base_model.share(rewrite_prompt)
        self.formatter = formatter

    def forward(self, query: str):
        res = self._m(query)
        if self.formatter == "list":
            return list(filter(None, res.split('\n')))
        else:
            return res

    def share(self, prompt: str = None):
        return QustionRewrite(self._m, rewrite_prompt=prompt, formatter=self.formatter)

    def status(self, task_name: Optional[str] = None):
        return self._m.status(task_name)

    @property
    def stream(self):
        return self._m._stream

    @stream.setter
    def stream(self, v: bool):
        self._m._stream = v

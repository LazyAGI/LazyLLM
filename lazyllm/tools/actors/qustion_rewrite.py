from typing import Union
from ...module import ModuleBase, TrainableModule


class QustionRewrite(ModuleBase):
    def __init__(
        self,
        base_model: Union[str, TrainableModule],
        rewrite_prompt: str = "",
        formatter: str = "str",
    ):
        super().__init__()
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(rewrite_prompt)
        else:
            self._m = base_model.share(rewrite_prompt)
        self.formatter = formatter

    def forward(self, query: str, **kw):
        res = self._m(query, **kw)
        if self.formatter == "list":
            return list(filter(None, res.split('\n')))
        else:
            return res

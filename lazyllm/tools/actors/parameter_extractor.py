from typing import Union, Optional
import json
from ...module import ModuleBase, TrainableModule

class ParameterExtractor(ModuleBase):
    type_map = {
        int.__name__: int,
        str.__name__: str,
        float.__name__: float,
        bool.__name__: bool,
    }

    def __init__(
        self,
        base_model: Union[str, TrainableModule],
        param: list[str],
        types: list[str],
        prompt: str = "",
    ):
        super().__init__()
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(prompt)
        else:
            self._m = base_model.share(prompt)
        self.param = param
        self.types = [ParameterExtractor.type_map[typename] for typename in types]

        self.param_dict = dict(zip(param, self.types))

    def forward(self, query: str):
        res = self._m(query)
        res = res.split("\n")
        ret = dict()
        for param in res:
            try:
                t = json.loads(param)
                for k, v in t.items():
                    if k in self.param:
                        ret[k] = None
                        if (
                            isinstance(v, self.param_dict[k])
                            and "__is_success" in t
                            and t["__is_success"] == 1
                        ):
                            ret[k] = v
                        break
            except Exception:
                continue
        return ret

    def share(self, prompt: str = ""):
        return ParameterExtractor(self._m, self.param, self.types, prompt=prompt)

    def status(self, task_name: Optional[str] = None):
        return self._m.status(task_name)

    @property
    def stream(self):
        return self._m._stream

    @stream.setter
    def stream(self, v: bool):
        self._m._stream = v

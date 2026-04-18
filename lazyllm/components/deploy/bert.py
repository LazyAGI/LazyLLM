import os
from typing import Any, Dict, Optional

import lazyllm
from lazyllm import LOG, LazyLLMLaunchersBase
from lazyllm.thirdparty import torch, transformers as tf

from .base import LazyLLMDeployBase
from .relay import RelayServer


class _BertSequenceClassificationService:

    def __init__(
        self,
        model_path: str,
        *,
        trust_remote_code: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
        init: bool = False,
    ):
        self._model_path = model_path
        self._trust_remote_code = trust_remote_code
        self._max_length = max_length
        self._device_pref = device
        self._tokenizer = None
        self._model = None
        self._device = None
        self._init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self._init_flag, self._load_model)

    def _load_model(self) -> None:
        self._tokenizer = tf.AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=self._trust_remote_code
        )
        self._model = tf.AutoModelForSequenceClassification.from_pretrained(
            self._model_path, trust_remote_code=self._trust_remote_code
        )
        if self._device_pref:
            self._device = torch.device(self._device_pref)
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)
        self._model.eval()
        LOG.info(f'Bert deploy: loaded model from {self._model_path} on {self._device}')

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lazyllm.call_once(self._init_flag, self._load_model)
        ta, tb = data.get('text_a'), data.get('text_b')
        text_a = ta.strip() if isinstance(ta, str) else ''
        text_b = tb.strip() if isinstance(tb, str) else ''
        if not text_a and not text_b:
            raise ValueError('At least one of text_a or text_b must be non-empty')

        if not text_b:
            enc = self._tokenizer(
                text_a,
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors='pt',
            )
        else:
            enc = self._tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors='pt',
            )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)

        pred = int(torch.argmax(probs, dim=-1).item())
        out = {
            'logits': logits.cpu().tolist()[0],
            'probs': probs.cpu().tolist()[0],
            'predicted_label': pred,
        }
        return out

    @classmethod
    def rebuild(cls, model_path, trust_remote_code, max_length, device, init):
        return cls(
            model_path,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            device=device,
            init=init,
        )

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self._init_flag)
        return _BertSequenceClassificationService.rebuild, (
            self._model_path,
            self._trust_remote_code,
            self._max_length,
            self._device_pref,
            init,
        )


class BertDeploy(LazyLLMDeployBase):

    message_format = {
        'text_a': '',
        'text_b': '',
    }
    keys_name_handle = {'inputs': 'text_a'}
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''
    stream_parse_parameters: dict = {}

    def __init__(
        self,
        launcher: Optional[LazyLLMLaunchersBase] = None,
        log_path: Optional[str] = None,
        trust_remote_code: bool = True,
        port: Optional[int] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        **kw,
    ):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port
        self._max_length = max_length
        self._device = device
        if kw:
            LOG.warning(f'Bert deploy: ignoring unknown kwargs: {sorted(kw.keys())}')

    def __call__(self, finetuned_model=None, base_model=None):
        finetuned_model = finetuned_model or ''
        base_model = base_model or ''
        if not finetuned_model:
            model_path = base_model
        else:
            valid_local = (
                os.path.isdir(finetuned_model)
                and any(
                    f.endswith(('.bin', '.safetensors', '.pt'))
                    for f in os.listdir(finetuned_model)
                )
            )
            if valid_local:
                model_path = finetuned_model
            elif base_model:
                LOG.warning(
                    f'Note! finetuned_model({finetuned_model}) is not a local checkpoint with weights; '
                    f'using base_model({base_model}).'
                )
                model_path = base_model
            else:
                LOG.info(
                    f'Bert deploy: finetuned_model({finetuned_model}) is not a valid local '
                    f'checkpoint and no base_model provided; treating it as a remote model id.'
                )
                model_path = finetuned_model

        if not model_path:
            raise ValueError('Bert deploy: finetuned_model and base_model are both empty or invalid.')

        func = _BertSequenceClassificationService(
            model_path,
            trust_remote_code=self._trust_remote_code,
            max_length=self._max_length,
            device=self._device,
        )
        return RelayServer(
            port=self._port,
            func=func,
            launcher=self._launcher,
            log_path=self._log_path,
            cls='bert',
        )()

    @staticmethod
    def extract_result(output: str, inputs) -> str:
        return output

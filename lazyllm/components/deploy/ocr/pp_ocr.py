import os
import lazyllm
from typing import Optional
import string
from ..base import LazyLLMDeployBase
from ...utils.file_operate import _base64_to_file

punctuation = set(string.punctuation + '，。！？；：“”‘’（）【】《》…—～、')


def is_all_punctuation(s: str) -> bool:
    return all(c in punctuation for c in s)


class _OCR(object):
    def __init__(
        self,
        model: Optional[str] = 'PP-OCRv5_mobile',
        use_doc_orientation_classify: Optional[bool] = False,
        use_doc_unwarping: Optional[bool] = False,
        use_textline_orientation: Optional[bool] = False,
        **kw
    ):
        self.model = model
        self.text_detection_model_name = model + '_det'
        self.text_recognition_model_name = model + '_rec'
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.init_flag = lazyllm.once_flag()

    def load_paddleocr(self):
        from lazyllm.thirdparty import paddleocr

        paddleocr_kwargs = {
            'use_doc_orientation_classify': self.use_doc_orientation_classify,
            'use_doc_unwarping': self.use_doc_unwarping,
            'use_textline_orientation': self.use_textline_orientation,
            'text_detection_model_name': self.text_detection_model_name,
            'text_recognition_model_name': self.text_recognition_model_name,
        }
        det_model_dir = os.path.join(lazyllm.config['model_path'], self.text_detection_model_name)
        if os.path.exists(det_model_dir):
            paddleocr_kwargs['det_model_dir'] = det_model_dir
        rec_model_dir = os.path.join(lazyllm.config['model_path'], self.text_recognition_model_name)
        if os.path.exists(rec_model_dir):
            paddleocr_kwargs['rec_model_dir'] = rec_model_dir

        self.ocr = paddleocr.PaddleOCR(**paddleocr_kwargs)

    def __call__(self, input):
        lazyllm.call_once(self.init_flag, self.load_paddleocr)
        if isinstance(input, dict):
            if 'inputs' in input:
                file_list = input['inputs']
        else:
            file_list = lazyllm.components.formatter.formatterbase._lazyllm_get_file_list(input)
        if isinstance(file_list, str):
            file_list = [file_list]
        file_list = [_base64_to_file(file) for file in file_list]
        if hasattr(file_list, '__repr__'):
            lazyllm.LOG.info(f'paddleocr read files: {file_list}')
        txt = []
        for file in file_list:
            if hasattr(self.ocr, 'predict'):
                result = self.ocr.predict(file)
            else:
                result = self.ocr.ocr(file)
            for res in result:
                for sentence in res['rec_texts']:
                    t = sentence.strip()
                    if not is_all_punctuation(t) and len(t) > 0:
                        txt.append(t)
        return '\n'.join(txt)

    @classmethod
    def rebuild(cls, *args, **kw):
        return cls(*args, **kw)

    def __reduce__(self):
        return _OCR.rebuild, (
            self.model,
            self.use_doc_orientation_classify,
            self.use_doc_unwarping,
            self.use_textline_orientation,
        )


class OCRDeploy(LazyLLMDeployBase):
    """OCRDeploy is a subclass of [LazyLLMDeployBase][lazyllm.components.LazyLLMDeployBase] that provides deployment for OCR (Optical Character Recognition) models.
This class is designed to deploy OCR models with additional configurations such as logging, trust for remote code, and port customization.

Attributes:

    keys_name_handle: A dictionary mapping input keys to their corresponding handler keys. For example:
        - "inputs": Handles general inputs.
        - "ocr_files": Also mapped to "inputs".
    message_format: A dictionary specifying the expected message format. For example:
        - {"inputs": "/path/to/pdf"} indicates that the model expects a PDF file path as input.
    default_headers: A dictionary specifying default headers for API requests. Defaults to:
        - {"Content-Type": "application/json"}

Args:
    launcher: A launcher instance for deploying the model. Defaults to `None`.
    log_path: A string specifying the path where logs should be saved. Defaults to `None`.
    trust_remote_code: A boolean indicating whether to trust remote code execution. Defaults to `True`.
    port: An integer specifying the port for the deployment server. Defaults to `None`.

Returns:
    OCRDeploy instance, can be started by calling


Examples:
    >>> from lazyllm.components import OCRDeploy
    >>> from lazyllm import launchers
    >>> # 创建一个 OCRDeploy 实例
    >>> deployer = OCRDeploy(launcher=launchers.local(), log_path='./logs', port=8080)
    >>> # 使用微调的 OCR 模型部署服务器
    >>> server = deployer(finetuned_model='ocr-model')
    >>> # 打印部署服务器信息
    >>> print(server)
    ... <RelayServer instance ready to handle OCR requests>
    """
    keys_name_handle = {
        'inputs': 'inputs',
        'ocr_files': 'inputs',
    }
    message_format = {'inputs': '/path/to/pdf'}
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, log_path=None, trust_remote_code=True, port=None):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(
            port=self._port, func=_OCR(finetuned_model), launcher=self._launcher, log_path=self._log_path, cls='ocr')()

import os
import lazyllm
from typing import Optional
import string
from ..base import LazyLLMDeployBase
from ...utils.file_operate import base64_to_file

punctuation = set(string.punctuation + "，。！？；：“”‘’（）【】《》…—～、")


def is_all_punctuation(s: str) -> bool:
    return all(c in punctuation for c in s)


class OCR(object):
    def __init__(
        self,
        model: Optional[str] = "PP-OCRv5_mobile",
        use_doc_orientation_classify: Optional[bool] = False,
        use_doc_unwarping: Optional[bool] = False,
        use_textline_orientation: Optional[bool] = False,
        **kw
    ):
        self.model = model
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.det_model_dir = model + "_det"
        self.rec_model_dir = model + "_rec"
        self.init_flag = lazyllm.once_flag()

    def load_paddleocr(self):
        from lazyllm.thirdparty import paddleocr

        paddleocr_kwargs = {
            'use_doc_orientation_classify': self.use_doc_orientation_classify,
            'use_doc_unwarping': self.use_doc_unwarping,
            'use_textline_orientation': self.use_textline_orientation,
        }
        if os.path.exists(self.det_model_dir):
            paddleocr_kwargs['det_model_dir'] = self.det_model_dir
        else:
            paddleocr_kwargs['text_detection_model_name'] = self.det_model_dir
        if os.path.exists(self.rec_model_dir):
            paddleocr_kwargs['rec_model_dir'] = self.rec_model_dir
        else:
            paddleocr_kwargs['text_recognition_model_name'] = self.rec_model_dir
        self.ocr = paddleocr.PaddleOCR(**paddleocr_kwargs)

    def __call__(self, input):
        lazyllm.call_once(self.init_flag, self.load_paddleocr)
        if isinstance(input, dict):
            if 'inputs' in input:
                file_list = input["inputs"]
        else:
            file_list = lazyllm.components.formatter.formatterbase._lazyllm_get_file_list(input)
        if isinstance(file_list, str):
            file_list = [file_list]
        file_list = [base64_to_file(file) for file in file_list]
        if hasattr(file_list, '__repr__'):
            lazyllm.LOG.info(f"paddleocr read files: {file_list}")
        txt = []
        for file in file_list:
            if hasattr(self.ocr, 'predict'):
                result = self.ocr.predict(file)
            else:
                result = self.ocr.ocr(file)
            for res in result:
                for sentence in res["rec_texts"]:
                    t = sentence.strip()
                    if not is_all_punctuation(t) and len(t) > 0:
                        txt.append(t)
        return "\n".join(txt)

    @classmethod
    def rebuild(cls, *args, **kw):
        return cls(*args, **kw)

    def __reduce__(self):
        return OCR.rebuild, (
            self.model,
            self.use_doc_orientation_classify,
            self.use_doc_unwarping,
            self.use_textline_orientation,
        )


class OCRDeploy(LazyLLMDeployBase):
    keys_name_handle = {
        "inputs": "inputs",
        "ocr_files": "inputs",
    }
    message_format = {"inputs": "/path/to/pdf"}
    default_headers = {"Content-Type": "application/json"}

    def __init__(self, launcher=None, log_path=None, trust_remote_code=True, port=None):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(
            port=self._port, func=OCR(finetuned_model), launcher=self._launcher, log_path=self._log_path, cls="ocr")()

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, cast
from fsspec import AbstractFileSystem
from lazyllm.thirdparty import PIL

from .readerBase import LazyLLMReaderBase, infer_torch_device
from ..doc_node import ImageDocNode

def img_2_b64(image: 'PIL.Image', format: str = "JPEG") -> str:
    buff = BytesIO()
    image.save(buff, format=format)
    return cast(str, base64.b64encode(buff.getvalue()))

def b64_2_img(data: str) -> 'PIL.Image':
    buff = BytesIO(base64.b64decode(data))
    return 'PIL.Image'.open(buff)

class ImageReader(LazyLLMReaderBase):
    def __init__(self, parser_config: Optional[Dict] = None, keep_image: bool = False, parse_text: bool = False,
                 text_type: str = "text", pytesseract_model_kwargs: Optional[Dict] = None,
                 return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._text_type = text_type
        if parser_config is None and parse_text:
            if text_type == "plain_text":
                try:
                    import pytesseract
                except ImportError:
                    raise ImportError("Please install extra dependencies that are required for the ImageReader "
                                      "when text_type is 'plain_text': `pip install pytesseract`")

                processor = None
                model = pytesseract
            else:
                try:
                    import sentencepiece  # noqa
                    import torch  # noqa
                    from PIL import Image  # noqa
                    from transformers import DonutProcessor, VisionEncoderDecoderModel
                except ImportError:
                    raise ImportError("Please install extra dependencies that are required for the "
                                      "ImageCaptionReader: `pip install torch transformers sentencepiece Pillow`")

                processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
                model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            parser_config = {'processor': processor, 'model': model}

        self._parser_config = parser_config
        self._keep_image = keep_image
        self._parse_text = parse_text
        self._pytesseract_model_kwargs = pytesseract_model_kwargs or {}

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[ImageDocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(path=file) as f:
                image = PIL.Image.open(f.read())
        else:
            image = PIL.Image.open(file)

        if image.mode != "RGB": image = image.convert("RGB")

        image_str: Optional[str] = None  # noqa
        if self._keep_image: image_str = img_2_b64(image)  # noqa

        text_str: str = ""
        if self._parse_text:
            assert self._parser_config is not None
            model = self._parser_config["model"]
            processor = self._parser_config["processor"]

            if processor:
                device = infer_torch_device()
                model.to(device)

                task_prompt = "<s_cord-v2>"
                decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False,
                                                        return_tensors='pt').input_ids
                pixel_values = processor(image, return_tensors='pt').pixel_values

                output = model.generate(pixel_values.to(device), decoder_input_ids=decoder_input_ids.to(device),
                                        max_length=model.decoder.config.max_position_embeddings, early_stopping=True,
                                        pad_token_id=processor.tokenizer.pad_token_id,
                                        eos_token_id=processor.tokenizer.eos_token_id, use_cache=True, num_beams=3,
                                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                        return_dict_in_generate=True)

                sequence = processor.batch_decode(output.sequences)[0]
                sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                text_str = re.sub(r"<.*?>", "", sequence, count=1).strip()
            else:
                import pytesseract

                model = cast(pytesseract, self._parser_config['model'])
                text_str = model.image_to_string(image, **self._pytesseract_model_kwargs)

        return [ImageDocNode(text=text_str, image_path=str(file), global_metadata=extra_info)]

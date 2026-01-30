# reader读取器的作用

负责对检索阶段获取的相关文档进行恰当的处理，以保证后面可以生成高质量的回答响应。

## 内置的reader类

- DocxReader:docx格式文件解析器，从 .docx 文件中读取文本内容并封装为文档节点（DocNode）列表。
- EpubReader:用于读取 .epub 格式电子书的文件读取器。当前版本不支持通过 fsspec 文件系统（如远程路径）加载 epub 文件，若提供 fs 参数，将回退到本地文件读取。
- HWPReader:HWP文件解析器，支持从本地文件系统读取 HWP 文件。它会从文档中提取正文部分的文本内容，返回 DocNode 列表。
- ImageReader:用于从图片文件中读取内容的模块。支持保留图片、解析图片中的文本（基于OCR或预训练视觉模型），并返回文本和图片路径的节点列表。
- IPYNBReader:用于读取和解析 Jupyter Notebook (.ipynb) 文件的模块。将 notebook 转换成脚本文本后，按代码单元划分为多个文档节点，或合并为单一文本节点。
- MinerUReader:基于Mineru服务的PDF解析器，通过调用Mineru服务的API来解析PDF文件，支持丰富的文档结构识别。
- MarkdownReader:用于读取和解析 Markdown 文件的模块。支持去除超链接和图片，按标题和内容将 Markdown 划分成若干文本段落节点。
- MboxReader:用于解析 Mbox 邮件存档文件的模块。读取邮件内容并格式化为文本，支持限制最大邮件数和自定义消息格式。
- TxtReader:用于从文本文件中加载内容，并将其封装为 DocNode 对象列表。
- PandasExcelReader:用于读取 Excel 文件（.xlsx），并将内容提取为文本。
- PDFReader:用于读取 PDF 文件并提取其中的文本内容。
- PPTXReader:用于解析 PPTX（PowerPoint）文件的读取器，能够提取幻灯片中的文本，并对嵌入图像进行视觉描述生成。
- VideoAudioReader:用于从视频或音频文件中提取语音内容的读取器，依赖 OpenAI 的 Whisper 模型进行语音识别。

## 基本使用方法

通过Document的add_reader添加指定文件阅读器

```python
from lazyllm.tools.rag.readers import MineruPDFReader

# 注册 PDF 解析器，url 替换为已启动的 MinerU 服务地址
documents.add_reader("*.pdf", MineruPDFReader(url="http://127.0.0.1:8888"))
```

手动调用解析方法

```python
from lazyllm.tools.rag import Document

doc = Document(dataset_path="your_doc_path")
data = doc._impl._reader.load_data(input_files=["your_doc_path/part_1.txt"])
```

## 自定义reade方法

定义解析函数，并注册到LazyLLM中

```python
from lazyllm.tools.rag import DocNode
from bs4 import BeautifulSoup

def processHtml(file, extra_info=None):
    text = ''
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        for element in soup.stripped_strings:
            text += element + '\n'
    node = DocNode(text=text, metadata=extra_info or {})
    return [node]

doc = Document(dataset_path="your_doc_path")
doc.add_reader("*.html", processHtml)
```

## 继承基类自定义reader

首先需要导入 ReaderBase 基类，然后自定义的 Reader 要继承自该基类，并重新实现 _load_data 接口，其他复杂的功能可以在类中其他的成员函数中实现即可。

注意：class 里面定义的 ​​_load_data​​ 接口和上面使用 Reader 时调用的 ​​load_data​​​ 接口完全没关系啊。class 里面的 ​​_load_data​​ 接口是每个 Reader 类中重载基类中的接口。而前面使用 Reader 时，调用的 ​​load_data​​ 接口是对文件列表进行遍历解析的接口。

```python
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag.readers.readerBase import infer_torch_device
from lazyllm.tools.rag import DocNode
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

class ImageDescriptionReader(ReaderBase):
    def __init__(self, parser_config: Optional[Dict] = None, prompt: Optional[str] = None) -> None:
        super().__init__()
        if parser_config is None:

            device = infer_torch_device()
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=dtype)
            parser_config = {"processor": processor, "model": model, "device": device, "dtype": dtype}
        self._parser_config = parser_config
        self._prompt = prompt

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        model = self._parser_config['model']
        processor = self._parser_config["processor"]

        device = self._parser_config["device"]
        dtype = self._parser_config["dtype"]
        model.to(device)

        inputs = processor(image, self._prompt, return_tensors="pt").to(device, dtype)

        out = model.generate(**inputs)
        text_str = processor.decode(out[0], skip_special_tokens=True)
        return [DocNode(text=text_str, metadata=extra_info or {})]
```

同样通过add_reader方法将这个读取器注册到对应document中

## 高性能开源工具MinerU的使用

```python
安装mineru依赖：lazyllm install mineru

启动部署：lazyllm deploy mineru [--port <port> [--cache_dir <cache_dir>] [--image_save_dir <image_save_dir>] [--model_source <model_source>]
```

参数说明:

|参数|说明|默认值|
|---------|---------|---------|
|--port|服务端口号|随机分配|
|--cache_dir|文档解析缓存目录（设置后相同文档无需重复解析）|None|
|--image_save_dir|图片输出目录（设置后保存文档内提取的图片）|None|
|--model_source|模型来源（可选：huggingface 或 modelscope）|huggingface|

使用示例:

```python
from lazyllm.tools.rag import Document
from lazyllm.tools.rag.readers import MineruPDFReader

doc = Document(dataset_path="your_doc_path")
# 注册 PDF 解析器，url 替换为已启动的 MinerU 服务地址
documents.add_reader("*.pdf", MineruPDFReader(url="http://127.0.0.1:8888"))
data = doc._impl._reader.load_data(input_files=["平安证券-珀莱雅.pdf"])
```

## 全局reader类的注册方式

通过register_global_reader方法注册的reader对于所有document可见

```python
from lazyllm.tools.rag import Document
from lazyllm.tools.rag.readers import MineruPDFReader

Document.register_global_reader("*.html", processHtml)
Document.register_global_reader("aa/*.html", HtmlReader)
Document.register_global_reader("aa/**/*.html", MineruPDFReader(url="http://127.0.0.1:8888"))  # url 需替换为已启动的 MinerU 服务地址
```

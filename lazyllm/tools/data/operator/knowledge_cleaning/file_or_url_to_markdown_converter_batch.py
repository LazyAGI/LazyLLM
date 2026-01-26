"""File or URL to Markdown Converter Batch operator"""
import os
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import requests
import pandas as pd
from lazyllm import LOG
from ...base_data import DataOperatorRegistry


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_pdf_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def download_pdf(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            pdf_folder = os.path.dirname(save_path)
            os.makedirs(pdf_folder, exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            LOG.info(f"PDF saved to {save_path}")
        else:
            LOG.warning("The URL did not return a valid PDF file.")
    except requests.exceptions.RequestException as e:
        LOG.error(f"Error downloading PDF: {e}")


def _parse_file_with_mineru(raw_file: str, output_file: str, mineru_backend: str = "vlm-vllm-engine") -> str:
    """Parse PDF/image files using MinerU"""
    try:
        import mineru  # noqa: F401
    except ImportError:
        raise Exception(
            "MinerU is not installed. Please refer to https://github.com/opendatalab/mineru to install."
        )

    os.environ['MINERU_MODEL_SOURCE'] = "local"
    MinerU_Version = {"pipeline": "auto", "vlm-transformers": "vlm", 'vlm-vllm-engine': 'vlm', 'vlm-http-client': 'vlm'}

    raw_file = Path(raw_file)
    pdf_name = Path(raw_file).stem
    intermediate_dir = os.path.join(output_file, "mineru")

    import subprocess
    command = [
        "mineru",
        "-p", str(raw_file),
        "-o", intermediate_dir,
        "-b", mineru_backend,
        "--source", "local"
    ]

    try:
        subprocess.run(command, check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to process file with MinerU: {str(e)}")

    PerItemDir = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend])
    output_file = os.path.join(PerItemDir, f"{pdf_name}.md")
    LOG.info(f"Markdown saved to: {output_file}")
    return output_file


def _parse_xml_to_md(raw_file: str = None, url: str = None, output_file: str = None):
    """Parse XML/HTML to Markdown using trafilatura"""
    try:
        from trafilatura import fetch_url, extract
    except ImportError:
        raise Exception("trafilatura is not installed. Please install it with 'pip install trafilatura'.")

    if url:
        downloaded = fetch_url(url)
        if not downloaded:
            downloaded = "fail to fetch this url. Please check your Internet Connection or URL correctness"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(downloaded)
            return output_file
    elif raw_file:
        with open(raw_file, "r", encoding='utf-8') as f:
            downloaded = f.read()
    else:
        raise Exception("Please provide at least one of file path and url string.")

    try:
        result = extract(downloaded, output_format="markdown", with_metadata=True)
        LOG.info(f"Extracted content is written into {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        LOG.error(f"Error during extract this file or link: {e}")

    return output_file


@DataOperatorRegistry.register(one_item=False, tag='knowledge_cleaning')
class FileOrURLToMarkdownConverterBatch:
    """
    Knowledge extractor supporting multiple file formats to Markdown conversion.
    知识提取算子：支持从多种文件格式中提取结构化内容并转换为标准Markdown。
    """

    def __init__(
            self,
            intermediate_dir: str = "intermediate",
            mineru_backend: str = "vlm-sglang-engine",
    ):
        self.intermediate_dir = intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.mineru_backend = mineru_backend

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识提取算子：支持从多种文件格式中提取结构化内容并转换为标准Markdown\n"
                "核心功能：\n"
                "1. PDF文件：使用MinerU解析引擎提取文本/表格/公式\n"
                "2. 网页内容(HTML/XML)：使用trafilatura提取正文\n"
                "3. 纯文本(TXT/MD)：直接透传不做处理"
            )
        else:
            return (
                "Knowledge Extractor: Converts multiple file formats to structured Markdown\n"
                "Key Features:\n"
                "1. PDF: Uses MinerU engine to extract text/tables/formulas\n"
                "2. Web(HTML/XML): Extracts main content using trafilatura\n"
                "3. Plaintext(TXT/MD): Directly passes through"
            )

    def __call__(
            self,
            data,
            input_key: str = "source",
            output_key: str = "text_path",
    ):
        """
        Convert files or URLs to Markdown.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input sources
            output_key: Key for output text paths

        Returns:
            List of dict with text paths added
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        LOG.info("Starting content extraction...")
        output_file_all = []

        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing files"):
            content = row.get(input_key, "")

            if is_url(content):
                if is_pdf_url(content):
                    pdf_save_path = os.path.join(self.intermediate_dir, f"crawled_{index}.pdf")
                    download_pdf(content, pdf_save_path)
                    content = pdf_save_path
                else:
                    output_file = os.path.join(self.intermediate_dir, f"crawled_{index}.md")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    output_file = _parse_xml_to_md(url=content, output_file=output_file)
                    output_file_all.append(output_file)
                    continue

            raw_file = content
            raw_file_name = os.path.splitext(os.path.basename(raw_file))[0]
            raw_file_suffix = os.path.splitext(raw_file)[1].lower()
            raw_file_suffix_no_dot = raw_file_suffix.lstrip(".")
            output_file = os.path.join(self.intermediate_dir, f"{raw_file_name}_{raw_file_suffix_no_dot}.md")

            if not os.path.exists(content):
                LOG.error(f"File not found: {content}")
                output_file_all.append("")
                continue

            ext = os.path.splitext(content)[1].lower()

            if ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                LOG.info(f"Using MinerU backend: {self.mineru_backend}")
                output_file = _parse_file_with_mineru(
                    raw_file=content,
                    output_file=self.intermediate_dir,
                    mineru_backend=self.mineru_backend
                )
            elif ext in [".html", ".xml"]:
                output_file = _parse_xml_to_md(raw_file=content, output_file=output_file)
            elif ext in [".txt", ".md"]:
                output_file = content
            else:
                LOG.error(f"Unsupported file type: {ext} for file {content}")
                output_file = ""

            output_file_all.append(output_file)

        dataframe[output_key] = output_file_all
        LOG.info("Content extraction completed!")
        return dataframe.to_dict('records')


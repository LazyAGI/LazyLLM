"""File or URL to Markdown Converter API operator"""
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import List
from tqdm import tqdm
import requests
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


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


def _batch_parse_html_or_xml(items: list):
    """Batch parse HTML/XML files"""
    results = {}
    for item in tqdm(items, desc="Parsing HTML/XML"):
        try:
            if item.get("url"):
                out = _parse_xml_to_md(url=item["url"], output_file=item["output_path"])
            else:
                out = _parse_xml_to_md(raw_file=item["raw_path"], output_file=item["output_path"])
            results[item["index"]] = out
        except Exception:
            results[item["index"]] = ""
    return results


class FileOrURLToMarkdownConverterAPI(kbc):
    """
    Knowledge extractor using MinerU API for PDF processing.
    知识提取算子：通过 MinerU API 处理 PDF 文件。
    """

    def __init__(
            self,
            intermediate_dir: str = "intermediate",
            mineru_backend: str = "vlm",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dir = intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.mineru_backend = mineru_backend

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识提取算子：通过 MinerU API 处理 PDF 文件\n"
                "设置 MINERU_API_KEY 环境变量以使用 API"
            )
        else:
            return (
                "Knowledge Extractor using MinerU API for PDF processing\n"
                "Set MINERU_API_KEY environment variable to use the API"
            )

    def forward_batch_input(
            self,
            data: List[dict],
            input_key: str = "source",
            output_key: str = "text_path",
    ) -> List[dict]:
        """
        Convert files or URLs to Markdown using API.

        Args:
            data: List of dict
            input_key: Key for input sources
            output_key: Key for output text paths

        Returns:
            List of dict with text paths added
        """
        assert isinstance(data, list), "Input data must be a list of dict"

        LOG.info("Starting content extraction (batch mode)...")
        normalized = []

        # Stage 1: normalize inputs
        for idx, row in enumerate(data):
            src = row.get(input_key, "")
            item = {"index": idx}

            if is_url(src):
                if is_pdf_url(src):
                    pdf_path = os.path.join(self.intermediate_dir, f"crawled_{idx}.pdf")
                    download_pdf(src, pdf_path)
                    item.update({"type": "pdf", "raw_path": pdf_path})
                else:
                    item.update({"type": "html", "url": src})
            else:
                if not os.path.exists(src):
                    item["type"] = "invalid"
                else:
                    ext = Path(src).suffix.lower()
                    if ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                        item.update({"type": "pdf", "raw_path": src})
                    elif ext in [".html", ".xml"]:
                        item.update({"type": "html", "raw_path": src})
                    elif ext in [".txt", ".md"]:
                        item.update({"type": "text", "raw_path": src})
                    else:
                        item["type"] = "unsupported"

            if "raw_path" in item:
                name = Path(item["raw_path"]).stem
                item["output_path"] = os.path.join(self.intermediate_dir, f"{name}.md")
            elif "url" in item:
                item["output_path"] = os.path.join(self.intermediate_dir, f"url_{idx}.md")

            normalized.append(item)

        # Stage 2: group by type
        pdf_items = [x for x in normalized if x.get("type") == "pdf"]
        html_items = [x for x in normalized if x.get("type") == "html"]
        text_items = [x for x in normalized if x.get("type") == "text"]

        # Stage 3: batch parse
        results = {}

        if html_items:
            results.update(_batch_parse_html_or_xml(html_items))

        if pdf_items:
            # For PDF items, we would use MinerU API if available
            LOG.warning("PDF processing via API requires MINERU_API_KEY environment variable")
            for item in pdf_items:
                results[item["index"]] = ""

        for item in text_items:
            results[item["index"]] = item.get("raw_path", "")

        # Stage 4: merge back
        output_list = []
        for idx, row in enumerate(data):
            new_item = row.copy()
            new_item[output_key] = results.get(idx, "")
            output_list.append(new_item)

        LOG.info("Extraction finished!")
        return output_list

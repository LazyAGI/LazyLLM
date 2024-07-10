from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests

from functools import partial
import re
from typing import Any, Callable, List, Tuple, Union

import nltk
import tiktoken

from .store import DocNode, MetadataMode
from lazyllm import LOG


def build_nodes_from_splits(
    text_splits: List[str], doc: DocNode, node_group: str
) -> List[DocNode]:
    nodes: List[DocNode] = []
    for text_chunk in text_splits:
        if not text_chunk:
            continue
        node = DocNode(
            text=text_chunk,
            ntype=node_group,
            embedding=doc.embedding,
            metadata=doc.metadata,
            excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
            parent=doc,
        )
        nodes.append(node)

    doc.children[node_group] = nodes
    return nodes


@dataclass
class _Split:
    text: str
    is_sentence: bool
    token_size: int


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text and keep the separator."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return result[1:] if len(result) > 0 and not result[0] else result


class NodeTransform(ABC):

    def forward(
        self, documents: Union[DocNode, List[DocNode]], node_group: str, **kwargs
    ) -> List[DocNode]:
        documents = documents if isinstance(documents, list) else [documents]
        all_nodes: List[DocNode] = []
        for node in documents:
            splits = self.transform(node, **kwargs)
            all_nodes.extend(build_nodes_from_splits(splits, node, node_group))
        return all_nodes

    @abstractmethod
    def transform(self, document: DocNode, **kwargs) -> List[str]:
        raise NotImplementedError("Not implemented")

    def __call__(
        self, nodes: List[DocNode], node_group: str, **kwargs: Any
    ) -> List[DocNode]:
        return self.forward(nodes, node_group, **kwargs)


class SentenceSplitter(NodeTransform):
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        assert (
            chunk_size > 0 and chunk_overlap >= 0
        ), "chunk size should > 0 and chunk_overlap should >= 0"

        try:
            self._tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except requests.exceptions.ConnectionError:
            LOG.error(
                "Unable to download the vocabulary file for tiktoken `gpt-3.5-turbo`. "
                "Please check your internet connection. "
                "Alternatively, you can manually download the file "
                "and set the `TIKTOKEN_CACHE_DIR` environment variable."
            )
            raise
        except Exception as e:
            LOG.error(f"Unable to build tiktoken tokenizer with error `{e}`")
            raise
        self._punkt_st_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

        self._sentence_split_fns = [
            partial(split_text_keep_separator, separator="\n\n\n"),  # paragraph
            self._punkt_st_tokenizer.tokenize,
        ]

        self._sub_sentence_split_fns = [
            partial(re.findall, pattern="[^,.;。？！]+[,.;。？！]?"),
            partial(split_text_keep_separator, separator=" "),
            list,  # split by character
        ]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def transform(self, document: DocNode, **kwargs) -> List[str]:
        return self.split_text(
            document.get_content(metadata_mode=MetadataMode.NONE),
            metadata_size=self._get_metadata_size(document),
        )

    def _get_metadata_size(self, node: DocNode) -> int:
        # Return the bigger size to ensure chunk_size < limit
        return max(
            self._token_size(node.get_metadata_str(mode=MetadataMode.EMBED)),
            self._token_size(node.get_metadata_str(mode=MetadataMode.LLM)),
        )

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        if text == "":
            return [""]
        effective_chunk_size = self.chunk_size - metadata_size
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_size}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            LOG.warning(
                f"Metadata length ({metadata_size}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        splits = self._split(text, effective_chunk_size)
        chunks = self._merge(splits, effective_chunk_size)
        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        """Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by paragraph separator
        2. split by chunking tokenizer
        3. split by second chunking regex
        4. split by default separator (" ")
        5. split by character
        """
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits_by_fns, is_sentence = self._get_splits_by_fns(text)

        text_splits = []
        for text in text_splits_by_fns:
            token_size = self._token_size(text)
            if token_size <= chunk_size:
                text_splits.append(
                    _Split(
                        text,
                        is_sentence=is_sentence,
                        token_size=token_size,
                    )
                )
            else:
                recursive_text_splits = self._split(text, chunk_size=chunk_size)
                text_splits.extend(recursive_text_splits)
        return text_splits

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        cur_chunk_len = 0
        is_chunk_new = True

        def close_chunk() -> None:
            nonlocal chunks, cur_chunk, cur_chunk_len, is_chunk_new

            chunks.append("".join([text for text, _ in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            is_chunk_new = True

            # Add overlap to the next chunk using the last one first
            overlap_len = 0
            for text, length in reversed(last_chunk):
                if overlap_len + length > self.chunk_overlap:
                    break
                cur_chunk.append((text, length))
                overlap_len += length
                cur_chunk_len += length
            cur_chunk.reverse()

        i = 0
        while i < len(splits):
            cur_split = splits[i]
            if cur_split.token_size > chunk_size:
                raise ValueError("Single token exceeded chunk size")
            if cur_chunk_len + cur_split.token_size > chunk_size and not is_chunk_new:
                # if adding split to current chunk exceeds chunk size
                close_chunk()
            else:
                if (
                    cur_split.is_sentence
                    or cur_chunk_len + cur_split.token_size <= chunk_size
                    or is_chunk_new  # new chunk, always add at least one split
                ):
                    # add split to chunk
                    cur_chunk_len += cur_split.token_size
                    cur_chunk.append((cur_split.text, cur_split.token_size))
                    i += 1
                    is_chunk_new = False
                else:
                    close_chunk()

        # handle the last chunk
        if not is_chunk_new:
            chunks.append("".join([text for text, _ in cur_chunk]))

        # Remove whitespace only chunks and remove leading and trailing whitespace.
        return [stripped_chunk for chunk in chunks if (stripped_chunk := chunk.strip())]

    def _token_size(self, text: str) -> int:
        return len(self._tiktoken_tokenizer.encode(text, allowed_special="all"))

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        for split_fn in self._sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

        for split_fn in self._sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False


class FuncNodeTransform(NodeTransform):
    """Used for user defined function.

    Wrapped the transform to: List[Docnode] -> List[Docnode]

    This wrapper supports:
        1. str -> list: transform=lambda t: t.split('\n')
        2. str -> str: transform=lambda t: t[:3]
    """

    def __init__(self, func: Callable[[str], List[str]]):
        self._func = func

    def transform(self, document: DocNode, **kwargs) -> List[str]:
        result = self._func(document.get_content(metadata_mode=MetadataMode.NONE))
        text_splits = [result] if isinstance(result, str) else result
        return text_splits

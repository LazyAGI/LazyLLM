from typing import List, Tuple
from .base import _TextSplitterBase, _Split, _UNSET
from typing import Optional, Union, AbstractSet, Collection, Literal, Any

class SentenceSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, chunk_overlap: int = _UNSET, num_workers: int = _UNSET):
        super().__init__(chunk_size=chunk_size, overlap=chunk_overlap, num_workers=num_workers)

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        return super().split_text(text, metadata_size)

    def from_tiktoken_encoder(self, encoding_name: str = 'gpt2', model_name: Optional[str] = None,
                              allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
                              disallowed_special: Union[Literal['all'], Collection[str]] = None,
                              **kwargs) -> 'SentenceSplitter':
        return super().from_tiktoken_encoder(encoding_name, model_name, allowed_special, disallowed_special, **kwargs)

    def from_huggingface_tokenizer(self, tokenizer: Any, **kwargs) -> 'SentenceSplitter':
        return super().from_huggingface_tokenizer(tokenizer, **kwargs)

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        cur_chunk_len = 0
        is_chunk_new = True

        def close_chunk() -> None:
            nonlocal cur_chunk, cur_chunk_len, is_chunk_new

            chunks.append(''.join([text for text, _ in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            is_chunk_new = True

            # Add overlap to the next chunk using the last one first
            overlap_len = 0
            for text, length in reversed(last_chunk):
                if overlap_len + length > self._overlap:
                    break
                cur_chunk.append((text, length))
                overlap_len += length
                cur_chunk_len += length
            cur_chunk.reverse()

        i = 0
        while i < len(splits):
            cur_split = splits[i]
            if cur_split.token_size > chunk_size:
                raise ValueError('Single token exceeded chunk size')
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
            chunks.append(''.join([text for text, _ in cur_chunk]))

        # Remove whitespace only chunks and remove leading and trailing whitespace.
        return [stripped_chunk for chunk in chunks if (stripped_chunk := chunk.strip())]

    def set_default(self, **kwargs):
        super().set_default(**kwargs)

    def get_default(self, param_name: Optional[str] = None):
        return super().get_default(param_name)

    def reset_default(self):
        super().reset_default()

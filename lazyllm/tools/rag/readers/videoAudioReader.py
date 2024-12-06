from pathlib import Path
from typing import Dict, List, Optional, cast
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class VideoAudioReader(LazyLLMReaderBase):
    def __init__(self, model_version: str = "base", return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._model_version = model_version

        try:
            import whisper
        except ImportError:
            raise ImportError("Please install OpenAI whisper model "
                              "`pip install git+https://github.com/openai/whisper.git` to use the model")

        model = whisper.load_model(self._model_version)
        self._parser_config = {"model": model}

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        import whisper

        if not isinstance(file, Path): file = Path(file)

        if file.name.endswith("mp4"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise ImportError("Please install pydub `pip install pydub`")

            if fs:
                with fs.open(file, 'rb') as f:
                    video = AudioSegment.from_file(f, format="mp4")
            else:
                video = AudioSegment.from_file(file, format="mp4")

            audio = video.split_to_mono()[0]
            file_str = str(file)[:-4] + ".mp3"
            audio.export(file_str, format="mp3")

        model = cast(whisper.Whisper, self._parser_config["model"])
        result = model.transcribe(str(file))

        transcript = result['text']
        return [DocNode(text=transcript, global_metadata=extra_info)]

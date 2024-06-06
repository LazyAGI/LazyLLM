from collections import defaultdict
import json
import os
import tempfile
from typing import Dict
import lazyllm

class FileHandlerBase:

    def __init__(self):
        self._roles = ["system", "knowledge", "user", "assistant"]

    def _validate_json(self, data_path: str) -> None:  # noqa C901
        # Check if file name format
        if os.path.splitext(data_path)[-1] != ".jsonl":
            raise ValueError("The file name must end with .jsonl")
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist.")

        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        # Initial dataset stats
        lazyllm.LOG.info("Num examples:", len(dataset))
        lazyllm.LOG.info("First example:")
        for message in dataset[0]["messages"]:
            lazyllm.LOG.info(message)

        # Format error checks
        format_error: Dict[str, list[int]] = defaultdict(list)
        for index, line in enumerate(dataset, start=1):
            # Check if example is a dictionary type
            if not isinstance(line, dict):
                format_error["data_type"].append(index)
                continue

            messages = line.get("messages", None)
            # Check if messages keyword exists
            if messages is None:
                format_error["missing_messages_list"].append(index)
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_error["message_missing_key"].append(index)

                if any(k not in ("role", "content") for k in message):
                    format_error["message_unrecognized_key"].append(index)

                if message.get("role", None) not in self._roles:
                    format_error["unrecognized_role"].append(index)

                content = message.get("content", None)
                if content is None or not isinstance(content, str):
                    format_error["missing_content"].append(index)

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_error["example_missing_assistant_message"].append(index)

        if format_error:
            lazyllm.LOG.error("Found errors: ")
            for k, v in format_error.items():
                lazyllm.LOG.error(f"Error Type: {k}, Error number: {len(v)}")
                lazyllm.LOG.error(f"Error Type: {k}, Error line number: {v}")
        else:
            lazyllm.LOG.info("No errors found")

    def get_finetune_data(self, filepath: str) -> str:
        self._validate_json(filepath)
        self._save_tempfile(self._convert_file_format(filepath))

    def _save_tempfile(self, data: str):
        self._dataHandler = tempfile.TemporaryFile()
        self._dataHandler.write(data.encode())
        self._dataHandler.seek(0)

    def _convert_file_format(self, filepath: str) -> str:
        raise NotImplementedError

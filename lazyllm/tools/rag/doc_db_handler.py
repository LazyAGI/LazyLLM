from typing import Union
from .document import Document
import weakref
from lazyllm.tools.sql.sql_manager import SqlManager
from lazyllm import OnlineChatModule, TrainableModule, LOG
from .doc_to_db import DocInfoSchema, DocToDbProcessor


class DocDbHandler:
    def __init__(self):
        self._sql_manager = None
        self._document: Document = None

    def post_init(self, document: Document, sql_manager: SqlManager):
        # use weakref to avoid circular reference
        self._document = weakref.ref(document)
        self._sql_manager = sql_manager
        self._doc_to_db_processor = DocToDbProcessor(sql_manager=sql_manager)

    def update(self, llm: Union[OnlineChatModule, TrainableModule]):
        file_paths = self._document._list_all_files_in_dataset()
        info_dicts = self._doc_to_db_processor.extract_info_from_docs(llm, file_paths)
        self._doc_to_db_processor.export_info_to_db(info_dicts)

    def extract_db_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], print_schema: bool = False
    ) -> DocInfoSchema:
        file_paths = self._document._list_all_files_in_dataset()
        self._doc_to_db_processor.analyze_info_schema_wo_genre_by_llm(llm, file_paths, apply_to_db=False)
        LOG.info(f"DocInfoSchema: {self._doc_to_db_processor.doc_info_schema}")
        return self._doc_to_db_processor.doc_info_schema

    def clear_and_reset_db_schema(self, db_info_schema: DocInfoSchema):
        self._doc_to_db_processor.reset_doc_info_schema(db_info_schema)

    def get_manager(self):
        if self._sql_manager is None:
            raise Exception("SqlManager is not initialized")
        return self._sql_manager

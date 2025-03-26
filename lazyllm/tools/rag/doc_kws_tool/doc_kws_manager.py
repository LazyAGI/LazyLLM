from typing import List
import lazyllm
from lazyllm import ModuleBase
from .doc_kws_prcoessor import DocTypeDetector, DocKWSExtractor, DocKWSGenerator, DocKwDesc, validate_kw_desc
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from sqlalchemy.orm import DeclarativeBase
import sqlalchemy
import uuid
from datetime import datetime


class TableBase(DeclarativeBase):
    pass


class DocKWSManager:

    DB_TYPE_MAP = {
        "int": sqlalchemy.Integer,
        "text": sqlalchemy.Text,
        "float": sqlalchemy.Float,
    }
    UUID_COL_NAME = "lazyllm_uuid"
    CREATED_AT_COL_NAME = "lazyllm_created_at"
    DOC_PATH_COL_NAME = "lazyllm_doc_path"

    def __init__(self, llm: ModuleBase, sql_manager: SqlManager, table_name="doc_kws"):
        self._doc_type_detector = DocTypeDetector(llm)
        self._kws_generator = DocKWSGenerator(llm)
        self._kws_extractor = DocKWSExtractor(llm)
        self._sql_manager = sql_manager
        self._kws_desc: List[DocKwDesc] = None
        self._table_name = table_name
        self._table_class = None
        self._doc_type = None

    @property
    def kws_desc(self):
        return self._kws_desc

    @kws_desc.setter
    def kws_desc(self, kws_desc: List[DocKwDesc]):
        raise NotImplementedError

    def clear(self):
        self._sql_manager.drop_table(self._table_name)
        del self._table_class
        self._doc_type = None
        self._table_class = None
        self._kws_desc = None

    def set_kws_desc(self, kws_desc: List[DocKwDesc]):
        assert isinstance(kws_desc, list)
        for kw_desc in kws_desc:
            is_success, err_msg = validate_kw_desc(kw_desc, DocKwDesc)
            assert is_success, err_msg
        self._kws_desc = kws_desc
        if self._table_class is not None:
            del self._table_class
        attrs = {"__tablename__": self._table_name, "__table_args__": {"extend_existing": True}}
        # use uuid as primary key
        attrs[self.UUID_COL_NAME] = sqlalchemy.Column(sqlalchemy.String(36), primary_key=True)
        attrs[self.CREATED_AT_COL_NAME] = sqlalchemy.Column(
            sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False
        )
        attrs[self.DOC_PATH_COL_NAME] = sqlalchemy.Column(
            sqlalchemy.Text, nullable=False, primary_key=False, index=True
        )
        for kw_desc in kws_desc:
            real_type = self.DB_TYPE_MAP.get(kw_desc["type"].lower(), sqlalchemy.Text)
            attrs[kw_desc["key"]] = sqlalchemy.Column(real_type, nullable=True, primary_key=False)
        self._table_class = type(self._table_name.capitalize(), (TableBase,), attrs)
        db_result = self._sql_manager.create_table(self._table_class)
        if db_result.status != DBStatus.SUCCESS:
            lazyllm.LOG.warning(f"Create table failed: {db_result.detail}")
            self.clear()

    def analyze_kws_desc(self, doc_type: str, doc_paths: List[str]) -> List[DocKwDesc]:
        return self._kws_generator.gen_kws_template(doc_type, doc_paths)

    def extract_kws_value(self, doc_path: str, extra_desc: str = ""):
        kws_value = self._kws_extractor.extract_kws_value(doc_path, self._kws_desc, extra_desc)
        if kws_value:
            kws_value[self.DOC_PATH_COL_NAME] = str(doc_path)
        else:
            lazyllm.LOG.warning(f"Extract kws value failed for {doc_path}")
        return kws_value

    def batch_extract_kws_value(self, doc_paths: List[str], extra_desc: str = ""):
        return [self.extract_kws_value(doc_path, extra_desc) for doc_path in doc_paths]

    def export_kws_values_to_db(self, kws_values: List[dict]):
        # Generate uuid explicitly because SQLite doesn't support auto gen uuid
        new_values = []
        for kws_value in kws_values:
            if kws_value:
                kws_value[self.UUID_COL_NAME] = str(uuid.uuid4())
                kws_value[self.CREATED_AT_COL_NAME] = datetime.now()
                new_values.append(kws_value)
        db_result = self._sql_manager.insert_values(self._table_name, new_values)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(f"Insert values failed: {db_result.detail}")

    def export_docs_kws_to_db(self, doc_paths: List[str], extra_desc: str = ""):
        kws_values = [self.extract_kws_value(doc_path, extra_desc) for doc_path in doc_paths]
        new_values = []
        for kws_value in kws_values:
            if kws_value:
                kws_value[self.UUID_COL_NAME] = str(uuid.uuid4())
                kws_value[self.CREATED_AT_COL_NAME] = datetime.now()
                new_values.append(kws_value)
        db_result = self._sql_manager.insert_values(self._table_name, new_values)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(f"Insert values failed: {db_result.detail}")

    def auto_export_docs_to_sql(self, doc_paths: List[str]):
        if len(doc_paths) == 0:
            return
        if self._doc_type is None:
            doc_type = self._doc_type_detector.get_doc_type(doc_paths[0])
            if doc_type == "":
                raise ValueError("Failed to detect doc type")
            self._doc_type = doc_type
        if self._kws_desc is None:
            kws_desc = self.analyze_kws_desc(self._doc_type, doc_paths)
            if not kws_desc:
                raise ValueError("Failed to detect kws_desc")
            self.set_kws_desc(kws_desc)
        self.export_docs_kws_to_db(doc_paths)

import os
from typing import List
from sqlalchemy import Column, Integer, String, DateTime, func, Enum as SQLEnum

from .db_manager import DBManager
from .db_operation import DBMergeClass, DBOperations
from .file_record import FileRecord
from enum import Enum

# Initialize the database
DB = DBManager.create_db("global.db")

class FileState(Enum):
    """
    Enumeration for file states.
    """
    WAIT_PARSE = "wait parse"    # Waiting to be parsed
    PENDING = "pending"          # Added to task queue, processing
    PARSED = "parsed"            # Successfully parsed
    PARSE_FAIL = "parse fail"    # Failed to parse
    WAIT_DELETE = "wait delete"  # Waiting to be deleted
    DELETED = "deleted"          # Successfully deleted

class KBFileRecord(DB.Base, DBOperations, metaclass=DBMergeClass, session=DB.session):
    """
    Model for knowledge files.
    """
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='Knowledge Base ID')
    kb_name = Column(String(50), comment='Knowledge Base Name')
    file_id = Column(String(36), comment='File ID')
    create_time = Column(DateTime, default=func.now(), comment='Creation Time')
    state = Column(SQLEnum(FileState), nullable=False, default=FileState.WAIT_PARSE)

    def __repr__(self):
        """
        String representation of the KBFileRecord instance.
        """
        return (f"<KBFileRecord(id='{self.id}', kb_name='{self.kb_name}', "
                f"state={self.state}, create_time='{self.create_time}')>")

    @classmethod
    def get_file_path_by_kb_name(cls, **kwargs) -> List[str]:
        """
        Get all file names and paths for a given knowledge base name.
        """
        with cls.session() as db_session:
            results = db_session.query(FileRecord.file_name, FileRecord.file_path)\
                                .join(KBFileRecord, KBFileRecord.file_id == FileRecord.id)\
                                .filter(kwargs)\
                                .all()
            files = [os.path.join(file_name, file_path) for file_name, file_path in results]
        return files
    
    @classmethod
    def get_file_id_by_kb_name(cls, kb_name: str) -> List[Integer]:
        """
        Get all file names and paths for a given knowledge base name.
        """
        with cls.session() as db_session:
            results = db_session.query(FileRecord.id)\
                                .join(KBFileRecord, KBFileRecord.file_id == FileRecord.id)\
                                .filter(KBFileRecord.kb_name == kb_name)\
                                .all()
        return results
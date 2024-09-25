from sqlalchemy import Column, Integer, String, DateTime, func, Enum

from .db_manager import DBManager
from .db_operation import DBMergeClass, DBOperations

# Initialize the database
DATABASE = DBManager.create_db("global.db")

FORDER_TYPE = "forder"

class FileRecord(DATABASE.Base, DBOperations, metaclass=DBMergeClass, session=DATABASE.session):
    """
    Represents a file record in the database.
    """
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, index=True, comment='Unique identifier for the file record')
    file_name = Column(String(255), nullable=False, comment='Name of the file')
    file_path = Column(String(255), nullable=False, comment='Path to the file')
    file_type = Column(Enum('TXT', 'PDF', 'XLSX','DOC','DOCX','JSON', 'forder', name='file_type'), nullable=False, comment='Type of the file')
    file_size = Column(Integer, nullable=False, comment='Size of the file in bytes')
    upload_time = Column(DateTime, default=func.now(), comment='Time when the file was uploaded')
    description = Column(String(255), nullable=True, comment='Description of the file')

    def __repr__(self):
        """
        String representation of the FileRecord instance.
        """
        return (f"<FileRecord(id={self.id}, file_name='{self.file_name}', "
                f"file_path='{self.file_path}', file_type='{self.file_type}', "
                f"file_size={self.file_size}, upload_time='{self.upload_time}', "
                f"description='{self.description}')>")

from sqlalchemy import Column, Integer, String, DateTime, func

from .db_manager import DBManager
from .db_operation import DBMergeClass, DBOperations

# Initialize the database
DB = DBManager.create_db("global.db")

class KBInfoRecord(DB.Base, DBOperations, metaclass=DBMergeClass, session=DB.session):
    """
    Model for knowledge base.
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='Knowledge Base ID')
    kb_name = Column(String(50), comment='Knowledge Base Name')
    kb_info = Column(String(200), comment='Knowledge Base Introduction (for Agent)')
    create_time = Column(DateTime, default=func.now(), comment='Creation Time')

    def __repr__(self):
        """
        String representation of the KBInfoRecord instance.
        """
        return (f"<KBInfoRecord(id='{self.id}', kb_name='{self.kb_name}', "
                f"kb_info='{self.kb_info}', create_time='{self.create_time}')>")
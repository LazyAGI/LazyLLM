from .db_manager import DBManager
from .db_operation import DBMergeClass, DBOperations
from sqlalchemy import Column, Integer, String

DB = DBManager.create_db()

class User(DB.Base, DBOperations, metaclass=DBMergeClass, session=DB.session):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, index=True, nullable=False)
    email = Column(String, index=True, nullable=False)

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', email='{self.email}')>"
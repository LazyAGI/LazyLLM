import os
import sys
curt_file_path = os.path.realpath(__file__) if "__file__" in globals() else os.getcwd()
sys.path.append(curt_file_path[:curt_file_path.index("LazyLLM") + len("LazyLLM")])

from .table_user import User
from .db_manager import DBManager

DBManager.create_db_tables()

user = User.create(username='alice', email='alice@example.com')
first = User.first(username="alice")
User.update(fun=lambda x: x.set(email='EMAIL'), username="alice")
item = User.first(username="alice")
tweets = User.all(username="alice")
for tweet in tweets:
    print(tweet)

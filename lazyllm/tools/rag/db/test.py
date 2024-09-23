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

def test_del_nodes(): # 删除
    User.create(username='a1', email='email_del_node')
    User.create(username='a2', email='email_del_node')
    User.create(username='a3', email='email_del_node')
    User.del_node(username=["a1","a2","a3"])
    assert User.all(username="email_del_node") == []
    
    User.create(username='a1', email='email_del_node')
    User.del_node(username='a1')
    assert User.all(username="a1") == []
    
def test_add_node(): # 添加节点
    User.del_node(username='add_node')
    user = User(username='add_node', email='-')
    User.add_node(user)
    assert User.first(username='add_node').email == '-'
    
def test_add_or_replace_node(): # 新增或替换节点
    User.del_node(username='add_node')
    User.del_node(email='email_add_or_replace_node')
    
    User.create(username='add_node', email='email_add_or_replace_node')
    assert User.first(email='email_add_or_replace_node').username == 'add_node'
    
    user = User(username='replace_node', email='email_add_or_replace_node')
    User.add_or_replace_node(user, email='email_add_or_replace_node')
    
    assert User.first(email='email_add_or_replace_node').username == 'replace_node'

def test_update(): # 更新节点
    User.del_node(username='update_node')
    User.del_node(email='email_update_node')
    
    User.create(username='update_node', email='email_update_node')
    User.update(fun=lambda node:node.set(email='email_new'), username='update_node')
    assert User.first(username='update_node').email == 'email_new'

def test_filter_order_by(): # orderby limit skip等功能
    User.del_node(username=["a1","a2","a3"])
    User.del_node(email='email_order')
    User.create(username='a3', email='email_order')
    User.create(username='a2', email='email_order')
    User.create(username='a1', email='email_order')
    
    nodes = User.filter_by(email='email_order')
    assert [node.username for node in nodes] == ['a3','a2','a1']
    
    nodes = User.filter_by(order_by = "username", email='email_order')
    assert [node.username for node in nodes] == ['a1','a2','a3']
    
    nodes = User.filter_by(limit=2, order_by = "username", email='email_order')
    assert len(nodes) == 2
    
    nodes = User.filter_by(skip=2, order_by = "username", email='email_order')
    assert len(nodes) == 1 and nodes[0].username == 'a3'
    
    

def test_multi_value_filter(): # 多值查询
    User.del_node(username=["a1","a2","a3"])
    User.create(username='a3', email='email_order')
    User.create(username='a2', email='email_order')
    User.create(username='a1', email='email_order')
    assert len(User.filter_by(username=["a1","a2","a3"])) == 3
    assert len(User.all(username=["a1","a2","a3"])) == 3


if __name__ == "__main__": 
    test_del_nodes()
    test_add_node()
    test_add_or_replace_node()
    test_update()
    test_filter_order_by()
    test_multi_value_filter()

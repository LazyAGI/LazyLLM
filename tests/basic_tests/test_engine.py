from lazyllm import LightEngine

nodes = [dict(id='123', kind='LocalModel', name='m1', args=dict(base_model='', deploy_method='dummy'))]
edges = [dict(iid='__start__', oid='123'), dict(iid='123', oid='__end__')]

engine = LightEngine()
engine.start(nodes, edges)
engine.run('1234')

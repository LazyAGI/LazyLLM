# Docuemnt的作用

Document 模块提供了统一的文档数据集管理接口，支持本地文件、云端文件或临时文档文件。它可以选择运行文档管理服务或 Web UI，并支持多种向量化模型和自定义存储后端。

## 参数

- dataset_path (Optional[str], default: None ) – 数据集目录路径。如果路径不存在，系统会尝试在 lazyllm.config["data_path"] 中查找。
- embed (Optional[Union[Callable, Dict[str, Callable]]], default: None ) – 文档向量化函数或函数字典。若为字典，键为 embedding 名称，值为对应的模型。
- manager (Union[bool, str], default: False ) – 是否启用文档管理服务。True 表示启动管理服务；'ui' 表示同时启动 Web 管理界面；默认 False。
- server (Union[bool, int], default: False ) – 是否为知识库运行服务接口。True 表示启动默认服务；整型数值表示自定义端口；False 表示关闭。默认为 False。
- name (Optional[str], default: None ) – 文档集合的名称标识符。默认为系统默认名称。
- launcher (Optional[LazyLLMLaunchersBase], default: None ) – 启动器实例，用于管理服务进程。默认使用远程异步启动器。
- store_conf (Optional[Dict], default: None ) – 存储配置。默认使用内存中的 MapStore。
- doc_fields (Optional[Dict[str, GlobalMetadataDesc]], default: None ) – 元数据字段配置，用于存储和检索文档属性。
- cloud (bool) – 是否为云端数据集。默认为 False。
- doc_files (Optional[List[str]], default: None ) – 临时文档文件列表。当使用此参数时，dataset_path 必须为 None，且仅支持 MapStore。
- processor (Optional[DocumentProcessor]) – 文档处理服务。
- display_name (Optional[str], default: '' ) – 文档模块的可读显示名称。默认为集合名称。
- description (Optional[str], default: 'algorithm description' ) – 文档集合的描述。默认为 "algorithm description"。

## Document基础使用示例

```python
import lazyllm
from lazyllm.tools import Document
m = lazyllm.OnlineEmbeddingModule(source="glm")
documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)
```

## 自定义存储后端

以mapstore和milvus作为例子，更多存储后端参考[Store存储使用示例](./store.md)

``` python
store_conf = {
    "segment_store": {
        "type": "map",
        "kwargs": {
            "uri": "/tmp/tmp_segments.db",
        },
    },
    "vector_store": {
        "type": "milvus",
        "kwargs": {
            "uri": "/tmp/tmp_milvus.db",
            "index_kwargs": {
                "index_type": "FLAT",
                "metric_type": "COSINE",
            },
        },
    },
}

# 添加额外需要存储的字段
doc_fields = {
    'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    'public_year': DocField(data_type=DataType.INT32),
}
document2 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, store_conf=store_conf, doc_fields=doc_fields)
```

## 指定文件读取器（可以自定义）

通过方法add_reader(pattern, func=None)添加读取器，作用范围仅对注册的 Document 对象可见。注册的文件读取器必须是 Callable 对象。只能通过函数调用的方式进行注册。并且通过实例注册的文件读取器的优先级高于通过类注册的文件读取器，并且实例和类注册的文件读取器的优先级高于系统默认的文件读取器。即优先级的顺序是：实例文件读取器 > 类文件读取器 > 系统默认文件读取器。

内置的读取器以及如何自定义读取器，可以参考：[Reader使用示例](./reader.md)

参数：
-pattern (str) – 文件读取器适用的匹配规则
-func (Callable, default: None ) – 文件读取器，必须是Callable的对象

```python
from lazyllm.tools.rag import Document, DocNode
from lazyllm.tools.rag.readers import ReaderBase
class YmlReader(ReaderBase):
...     def _load_data(self, file, fs=None):
...         try:
...             import yaml
...         except ImportError:
...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
...         with open(file, 'r') as f:
...             data = yaml.safe_load(f)
...         print("Call the class YmlReader.")
...         return [DocNode(text=data)]
...
def processYml(file):
...     with open(file, 'r') as f:
...         data = f.read()
...     print("Call the function processYml.")
...     return [DocNode(text=data)]
...
doc1 = Document(dataset_path="your_files_path", create_ui=False)
doc2 = Document(dataset_path="your_files_path", create_ui=False)
doc1.add_reader("**/*.yml", YmlReader)
print(doc1._impl._local_file_reader)
{'**/*.yml': <class '__main__.YmlReader'>}
print(doc2._impl._local_file_reader)
{}
files = ["your_yml_files"]
Document.register_global_reader("**/*.yml", processYml)
doc1._impl._reader.load_data(input_files=files)
Call the class YmlReader.
doc2._impl._reader.load_data(input_files=files)
Call the function processYml.
```

此外可以通过register_global_reader注册读取器，作用范围对于所有的 Document 对象都可见。

## 创建节点和组

创建知识库分组

- create_kb_group(name, doc_fields=None, store_conf=None)
创建一个新的知识库分组（KB Group），并返回绑定到该分组的文档对象。
知识库分组用于在同一个文档模块中划分不同的文档集合，每个分组可以有独立的字段定义和存储配置。

参数:

- name (str) – 知识库分组的名称。
- doc_fields (Optional[Dict[str, GlobalMetadataDesc]], default: None ) – 文档字段定义。指定每个字段的名称、类型和描述。
- store_conf (Optional[Dict], default: None ) – 存储配置，用于定义存储后端及其参数。

创建节点分组

- create_node_group(name=None, *, transform, parent=LAZY_ROOT_NAME, trans_node=None, num_workers=0, display_name=None, group_type=NodeGroupType.CHUNK, **kwargs)
创建一个由指定规则生成的 node group。

参数:

- name (str, default: None ) – node group 的名称。
- transform (Callable) – 将 node 转换成 node group 的转换规则，函数原型是 (DocNode, group_name, **kwargs) -> List[DocNode]。目前内置的有 SentenceSplitter。用户也可以自定义转换规则。
- trans_node (bool, default: None ) – 决定了transform的输入和输出是 DocNode 还是 str ，默认为None。只有在 transform 为 Callable 时才可以设置为true。
- num_workers (int, default: 0 ) – Transform时所用的新线程数量，默认为0
- parent (str, default: LAZY_ROOT_NAME ) – 需要进一步转换的节点。转换之后得到的一系列新的节点将会作为该父节点的子节点。如果不指定则从根节点开始转换。
- kwargs – 和具体实现相关的参数。

```python
import lazyllm
from lazyllm.tools import Document, SentenceSplitter
m = lazyllm.OnlineEmbeddingModule(source="glm")
documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
```

更多transform方式，可以参考[Transform使用示例](./transform.md)

## 激活分组

- activate_group(group_name, embed_keys=None, enable_embed=True)
激活指定的知识库分组，并可选择指定要启用的 embedding key。
激活后，文档模块会在该分组下执行检索和存储操作。如果未指定 embedding key，则默认启用所有可用的 embedding。

参数：

- group_name (str) – 要激活的知识库分组名称。
- embed_keys (Optional[Union[str, List[str]]], default: None ) – 需要启用的 embedding key，可以是单个字符串或字符串列表。默认为空列表，表示启用全部 embedding。

- activate_groups(groups, **kwargs)
批量激活多个知识库分组。
该方法会依次调用 activate_group 来激活传入的所有分组。

参数:

groups (Union[str, List[str]]) – 要激活的分组名称或分组名称列表。

## 获取节点列表

- get_nodes(uids=None, doc_ids=None, group=None, kb_id=None, numbers=None)

参数:

- uids (Optional[List[str]], default: None ) – 指定节点 uid 列表。
- doc_ids (Optional[Set], default: None ) – 指定文档 id 集合。
- group (Optional[str], default: None ) – 节点组名。
- kb_id (Optional[str], default: None ) – 知识库 id。
- numbers (Optional[Set], default: None ) – 节点编号集合。

``` python
import lazyllm
from lazyllm.tools import Document
doc = Document()
nodes = doc.get_nodes(doc_ids={'doc_1'}, group='CoarseChunk', kb_id='kb_1', numbers={1, 2})
```

## 离在线分离，接入远程部署的Document
Document支持远程部署与接入功能，支持用户将Document部署在远程服务器上，并使用url的方式接入。

在远程服务上启动Document
```python
# 使用服务模式启动Document
docs = lazyllm.Document(dataset_path="rag_master",
                        name="doc_server",
                        embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                        server=9200,
                        store_conf=milvus_store_conf,
                        doc_fields=doc_fields)
docs.create_node_group(name="sentences", transform=lambda s: '。'.split(s))
docs.activate_groups(["sentences", "CoarseChunk"]) # 使用 docs.activate_group('sentences', embed_keys=['xxx']) 激活单个节点组

docs.start()
```

启动后，假设文档服务部署在 127.0.0.1 的 9200 端口，则可以通过 http://127.0.0.1:9200/ 访问文档服务。我们使用 lazyllm.UrlDocument 来接入文档服务，并指定文档服务的名称 doc_server。
```python
# 使用url接入Document
docs2 = lazyllm.Document(url="http://127.0.0.1:9200/", name="doc_server")
retriever = lazyllm.Retriever(doc=docs2, group_name="sentences", topk=3)

query = "何为天道？"
res = retriever(query=query)
print(f"answer: {res}")
```



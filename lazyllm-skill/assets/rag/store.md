# Store的作用

Store负责在Node转换之后，将转换得到的 Node Group 内容保存起来，这样后续使用的时候可以避免重复执行转换操作。

## 内置的Store类

混合存储类型的存储后端支持Segment和Vector，切片存储类型只支持Segment，向量存储只支持Vector

### Hybrid Store（混合存储类型）

#### Mapstore

基于SQLite的Map存储类，提供基于SQLite的数据持久化与BM25全文检索，支持多集合管理和简单查询。

参数：

- uri (Optional[str], default: None ) – SQLite数据库文件路径，默认为None（内存模式）
- **kwargs – 其他关键字参数

#### OceanBaseStore

基于OceanBase和seekdb的存储类，用于存储和检索文档节点。

参数：

- uri (str, default: '127.0.0.1:2881' ) – OceanBase 数据库的 URI。
- user (str) – OceanBase 数据库的用户名。
- password (str) – OceanBase 数据库的密码。
- db_name (str, default: 'test' ) – OceanBase 数据库的名称。
- drop_old (bool) – 是否删除旧的表。
- index_kwargs (List[dict], default: None ) – 索引配置列表。
- client_kwargs (Dict, default: None ) – 客户端配置字典。
- max_pool_size (int) – 最大连接池大小。
- normalize (bool) – 是否规范化数据。
- enable_fulltext_index (bool) – 是否启用全文索引。

#### SenseCoreStore

### Segment Store（切片存储类型）

#### OpenSearchStore
OpenSearch存储类，提供基于OpenSearch的文档存储和检索功能，支持大规模文档管理和高效查询。

参数：

- uris (List[str]) – OpenSearch服务URI列表
- client_kwargs (Optional[Dict], default: None ) – OpenSearch客户端配置参数
- index_kwargs (Optional[Union[Dict, List]], default: None ) – 索引配置参数
- **kwargs – 其他关键字参数

#### ElasticSearchStore

基于 Elasticsearch 的向量存储实现，继承自 StoreBase。支持向量写入、删除、相似度检索，兼容标量过滤。

参数：

- uris (List[str]): Elasticsearch 连接 URI（如 ["http://localhost:9200"]）。
- client_kwargs (Optional[Dict]): 传递给 Elasticsearch 客户端的额外参数。
- index_kwargs (Optional[Union[Dict, List]]): 索引创建参数（例如 {"index_type": "IVF_FLAT", "metric_type": "CONSINE"} ，支持按向量模型的key配置列表）。
- **kwargs: 预留扩展参数。

### Vector Store（向量存储类型）

#### ChromaStore

ChromaStore 是基于 Chroma 的向量存储实现，支持向量写入、检索与持久化。

参数：

- uri (Optional[str], default: None ) – Chroma 连接 URI，当未指定 dir 时必填。
- dir (Optional[str], default: None ) – 本地持久化存储路径，提供时使用 PersistentClient 模式。
- index_kwargs (Optional[Union[Dict, List]], default: None ) – Collection 配置参数，如索引类型、距离度量方式等。
- client_kwargs (Optional[Dict], default: None ) – 传递给 Chroma 客户端的额外参数。
- **kwargs – 预留扩展参数。

#### MilvusStore

基于 Milvus 的向量存储实现，支持向量写入、删除、相似度检索，兼容标量过滤。

参数：

- uri (str, default: '' ) – Milvus 连接 URI（如 "tcp://localhost:19530"）。如果为本地路径则使用milvus-lite，否则为远程模式（需要独立部署milvus服务，例如standalone/distributed版本）。
- db_name (str, default: 'lazyllm' ) – Milvus 中使用的数据库名称，默认为 "lazyllm"。
- index_kwargs (Optional[Union[Dict, List]], default: None ) – 索引创建参数（例如 {"index_type": "IVF_FLAT", "metric_type": "CONSINE"} ，支持按向量模型的key配置列表）。
- client_kwargs (Optional[Dict], default: None ) – 传递给 milvus 客户端的额外参数。

## 基础使用

###配置单一存储的store_conf
```python
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': '/path/to/milvus/dir/milvus.db',
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}
```

###配置多个存储的store_conf
```python
store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'milvus',
        'kwargs': {
            'uri': '/path/to/milvus/dir/milvus.db',
            'index_kwargs': {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            }
        },
    },
}

doc = Document(dataset_path="/path/to/your/doc/dir",
               embed=lazyllm.OnlineEmbeddingModule(),
               store_conf = store_conf)
```

# Store的作用

Store负责在Node转换之后，将转换得到的 Node Group 内容保存起来，这样后续使用的时候可以避免重复执行转换操作。

## 内置的Store类

混合存储类型的存储后端支持Segment和Vector，切片存储类型只支持Segment，向量存储只支持Vector

### Hybrid Store（混合存储类型）

#### Mapstore

基于SQLite的Map存储类，提供基于SQLite的数据持久化与BM25全文检索，支持多集合管理和简单查询。

store_conf配置:

1. 如果将切片存储在本地路径/tmp/mapstore.db中，配置如下：

store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/tmp/mapstore.db',  # 指定本地存储路径
        },
    },
}

2. 如果用纯内存存储（不持久化到本地文件），可以省略uri参数：

store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {},
    },
}

#### OceanBaseStore

基于OceanBase和seekdb的存储类，用于存储和检索文档节点。

store_conf = {
    'type': 'oceanbase',  # 使用OceanBase作为向量存储
    'kwargs': {
        'uri': '127.0.0.1:2881',  # OceanBase服务器地址
        'user': 'root@test',  # 用户名
        'password': '',  # 密码
        'db_name': 'test',  # 数据库名称
        'index_kwargs': [
            {
                'embed_key': 'vec_dense',  # 向量键
                'index_type': 'FLAT',  # 索引类型
                'metric_type': 'COSINE',  # 距离度量
            },
            {
                'embed_key': 'vec_sparse',  # 向量键
                'index_type': 'HNSW',  # 索引类型
                'metric_type': 'L2',  # 距离度量
            },
        ],
    },
    'global_metadata_desc': {
        'RAG_KB_ID': 'kb_id',  # 全局元数据键
        'RAG_DOC_ID': 'doc_id',  # 全局元数据键
    },
}

OceanBase支持的字段详细参考: [oceanbase官方文档](https://www.oceanbase.com/docs/oceanbase-database-cn)

#### SenseCoreStore

基于SenseCore的存储类，提供基于SenseCore的文档存储和检索功能，支持大规模文档管理和高效查询。

SenseCore的使用手册参考: [SenseCore官方文档](https://www.sensecore.cn/help#storage)

### Segment Store（切片存储类型）

#### OpenSearchStore
OpenSearch存储类，提供基于OpenSearch的文档存储和检索功能，支持大规模文档管理和高效查询。

store_conf配置:

store_conf = {
    'type': 'opensearch',
    'kwargs': {
        'uris': 'https://127.0.0.1:9200',  # OpenSearch 服务地址
        'client_kwargs': {
            'http_compress': True,        # 是否启用 HTTP 压缩
            'use_ssl': True,              # 是否使用 SSL
            'verify_certs': False,        # 是否验证 SSL 证书
            'user': 'admin',              # 用户名
            'password': 'admin_password'  # 密码
        },
        'index_kwargs': {
            'index_name': 'my_index',     # 索引名称
            'settings': {
                'number_of_shards': 1,    # 分片数量
                'number_of_replicas': 1   # 副本数量
            }
        }
    }
}

OpenSearch支持的字段详细参考: [opensearch官方文档](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)

#### ElasticSearchStore

基于 Elasticsearch 的切片存储实现。支持向量写入、删除、相似度检索，兼容标量过滤。

store_conf配置:

store_conf = {
    'type': 'elasticsearch',
    'kwargs': {
        'uris': 'https://127.0.0.1:9200',  # Elasticsearch 服务地址
        'client_kwargs': {
            'http_compress': True,        # 是否启用 HTTP 压缩
            'use_ssl': True,              # 是否使用 SSL
            'verify_certs': False,        # 是否验证 SSL 证书
            'user': 'elastic',            # 用户名
            'password': 'password'        # 密码
        },
        'index_kwargs': {
            'index_name': 'my_index',     # 索引名称
            'settings': {
                'number_of_shards': 1,    # 分片数量
                'number_of_replicas': 1   # 副本数量
            }
        }
    }
}

ElasticSearch支持的字段详细参考: [elasticsearch官方文档](https://www.elastic.co/docs/api/doc/elasticsearch/)

### Vector Store（向量存储类型）

#### ChromaStore

ChromaStore 是基于 Chroma 的向量存储实现，支持向量写入、检索与持久化。

store_conf配置:

store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'chroma',
        'kwargs': {
            'dir': '/path/to/vector/dir',
            'index_kwargs': {
                'hnsw': {
                    'space': 'cosine',
                    'ef_construction': 200,
                }
            }
        },
    },
}

Chroma支持的字段详细参考: [chroma官方文档](https://docs.trychroma.com/)

#### MilvusStore

基于 Milvus 的向量存储实现，支持向量写入、删除、相似度检索，兼容标量过滤。

store_conf配置:

store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'chroma',
        'kwargs': {
            'dir': '/path/to/vector/dir',
            'index_kwargs': [
                {
                    'embed_key': 'vec1',
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                },
                {
                    'embed_key': 'vec2',
                    'index_type': 'SPARSE_INVERTED_INDEX',
                    'metric_type': 'IP',
                }
            ]
        },
    },
}

Milvus支持的字段详细参考: [milvus官方文档](https://milvus.io/docs/v2.3.0/index.md)

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

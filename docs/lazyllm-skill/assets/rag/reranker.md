# Reranker的作用

通过重排序流程，可以对其进行更精细的排序，得到排序后的文档，使得更重要的文档在返回列表中更靠前。

## 参数

- name (str, default: 'ModuleReranker' ) ： 实现重排序时必须为 ModuleReranker
- model（Union[Callable, str]）： 实现重排序的具体模型名称或可调用对象
    * Callable 情形：
        OnlineEmbeddingModule (type="rerank")：目前支持qwen和glm的在线重排序模型，使用前需要指定 apikey
        TrainableModule(model="str")：需要传入本地模型名称，常用的开源重排序模型为bge-reranker系列
    * str 情形：模型名称，与上 Callable 情形 TrainableModule 对应的 model 参数要求相同
- topk（int）： 最终需要返回的 k 个节点数
- output_format（str, default: None）：输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典
- join（boolean, default: False）：是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。

## 基本使用

```python
reranker = lazyllm.Reranker(
    name='ModuleReranker',
    model=lazyllm.OnlineEmbeddingModule(type="rerank"),
    topk=1
)

# 对多个检索结果排序
doc_node_list = reranker(
    nodes=doc_node_list_1 + doc_node_list_2,
    query="用户问题"
)
```

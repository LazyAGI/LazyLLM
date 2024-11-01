import os
import concurrent
from typing import Dict, Callable, List
from lazyllm import config, ThreadPoolExecutor
from .doc_node import DocNode

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    "max_embedding_workers",
    int,
    min(32, (os.cpu_count() or 1) + 4),
    "MAX_EMBEDDING_WORKERS",
)

# returns a list of modified nodes
def parallel_do_embedding(embed: Dict[str, Callable], nodes: List[DocNode]) -> List[DocNode]:
    modified_nodes = []
    with ThreadPoolExecutor(config["max_embedding_workers"]) as executor:
        futures = []
        for node in nodes:
            miss_keys = node.has_missing_embedding(embed.keys())
            if not miss_keys:
                continue
            modified_nodes.append(node)
            for k in miss_keys:
                with node._lock:
                    if node.has_missing_embedding(k):
                        future = executor.submit(node.do_embedding, {k: embed[k]}) \
                            if k not in node._embedding_state else executor.submit(node.check_embedding_state, k)
                        node._embedding_state.add(k)
                        futures.append(future)
        if len(futures) > 0:
            for future in concurrent.futures.as_completed(futures):
                future.result()
    return modified_nodes

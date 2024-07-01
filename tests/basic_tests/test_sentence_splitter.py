from lazyllm.tools.rag.parser import SentenceSplitter
from lazyllm.tools.rag.store import DocNode, MetadataMode


class TestSentenceSplitter:
    def setup_method(self):
        """Setup for tests: initialize the SentenceSplitter."""
        self.splitter = SentenceSplitter(chunk_size=100, chunk_overlap=30)

    def test_forward(self):
        text = """ Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.
        第1章 引言 人工智能（AI）是计算机科学的一个分支，旨在模拟和增强人类智能。第2章 历史 AI 的发展可以追溯到20世纪50年代，当时计算机科学家开始研究如何让机器具有人类智能。第3章 应用 AI 技术在各个领域都有应用，包括医疗、金融、制造和交通等。第3.1节 医疗 在医疗领域，AI 可以用于疾病诊断、治疗方案推荐和患者监控。第3.2节 金融 在金融领域，AI 可以用于风险评估、股票交易和客户服务。第4章 未来展望 随着技术的进步，AI 的应用将越来越广泛，可能会彻底改变我们的生活方式。
        """
        docs = [DocNode(text=text)]

        result = self.splitter.forward(docs)
        result_texts = [n.get_content(metadata_mode=MetadataMode.NONE) for n in result]
        expected_texts = [
            "Before college the two main things I worked on, outside of school, were writing and programming.I didn't write essays.I wrote what beginning writers were supposed to write then, and probably still are: short stories.My stories were awful.They had hardly any plot, just characters with strong feelings, which I imagined made them deep.",
            "My stories were awful.They had hardly any plot, just characters with strong feelings, which I imagined made them deep.第1章 引言 人工智能（AI）是计算机科学的一个分支，旨在模拟和增强人类智能。",
            "第2章 历史 AI 的发展可以追溯到20世纪50年代，当时计算机科学家开始研究如何让机器具有人类智能。第3章 应用 AI 技术在各个领域都有应用，包括医疗、金融、制造和交通等。",
            "第3.1节 医疗 在医疗领域，AI 可以用于疾病诊断、治疗方案推荐和患者监控。第3.2节 金融 在金融领域，AI 可以用于风险评估、股票交易和客户服务。",
            "第4章 未来展望 随着技术的进步，AI 的应用将越来越广泛，可能会彻底改变我们的生活方式。",
        ]
        assert (
            result_texts == expected_texts
        ), f"Expected {expected_texts}, but got {result_texts}"

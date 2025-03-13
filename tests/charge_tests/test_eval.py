import lazyllm
from lazyllm.tools.eval import ResponseRelevancy, Faithfulness

class TestEvalRAG:

    def setup_method(self):
        self.data = [{
            'question': '非洲的猴面包树果实的长度约是多少厘米？',
            'answer': '非洲猴面包树的果实长约15至20厘米。它非常的长。',
            'context': (
                '非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。'
                '钙含量比菠菜高50％以上，含较高的抗氧化成分，维生素C含量是单个橙子的三倍。')
        }]

    def test_response_relevancy(self):
        m = ResponseRelevancy(
            lazyllm.OnlineChatModule(),
            lazyllm.OnlineEmbeddingModule())
        res = m(self.data)
        assert isinstance(res, float)

    def test_faithfulness(self):
        m = Faithfulness(lazyllm.OnlineChatModule())
        res = m(self.data)
        assert isinstance(res, float)

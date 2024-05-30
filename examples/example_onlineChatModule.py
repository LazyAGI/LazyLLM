import unittest
import lazyllm

class TestOnlineChatModule(unittest.TestCase):

    def test_openai_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if len(r['content']) > 0:
                    content += r['content']
                print(f"response: {r['content']}")
            history.append([query, content])
            print(f"history: {history}")

    def test_openai_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_openai_finetune(self):
        m = lazyllm.OnlineChatModule(source="openai", model="gpt-3.5-turbo-0125", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_kimi_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="kimi", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if len(r['content']) > 0:
                    content += r['content']
                print(f"response: {r['content']}")
            history.append([query, content])
            print(f"history: {history}")

    def test_kimi_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="kimi", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_glm_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="glm", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if len(r['content']) > 0:
                    content += r['content']
                print(f"response: {r['content']}")
            history.append([query, content])
            print(f"history: {history}")

    def test_glm_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="glm", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_glm_finetune(self):
        m = lazyllm.OnlineChatModule(source="glm", model="chatglm3-6b", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_qwen_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="qwen", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if len(r['content']) > 0:
                    content += r['content']
                print(f"response: {r['content']}")
            history.append([query, content])
            print(f"history: {history}")

    def test_qwen_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="qwen", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_qwen_finetune(self):
        m = lazyllm.OnlineChatModule(source="qwen", model="qwen-turbo", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_sensenova_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="sensenova", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if len(r['content']) > 0:
                    content += r['content']
                print(f"response: {r['content']}")
            history.append([query, content])
            print(f"history: {history}")

    def test_sensenova_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="sensenova", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        import time
        import random
        for query in querys:
            time.sleep(random.randint(2, 5))
            resp = m(query, llm_chat_history=history, max_new_tokens=20)
            history.append([query, resp["content"]])
            print(f"history: {history}")

    def test_sensenova_finetune(self):
        m = lazyllm.OnlineChatModule(source="sensenova", model="nova-ptc-s-v2", stream=False)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file, upload_url="https://file.sensenova.cn/v1/files")
        m._get_train_tasks()
        m._get_deploy_tasks()


if __name__ == '__main__':
    unittest.main()

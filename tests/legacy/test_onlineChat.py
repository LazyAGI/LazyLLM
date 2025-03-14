import unittest
import lazyllm
import json

def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})
    
def get_n_day_weather_forecast(location, num_days, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "unit"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "unit", "num_days"]
            },
        }
    },
]

prompter = lazyllm.ChatPrompter(instruction="Answer the following questions as best as you can. You have access to the following tools:\n", 
                                extra_keys=["tools"], 
                                show=True)

class TestOnlineChatModule(unittest.TestCase):

    def test_openai_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                    delta = r["choices"][0]["delta"]
                    content += delta["content"]
                print(f"response: {content}")
            history.append([query, content])
            print(f"history: {history}")

    def test_openai_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            resp = resp['message']
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_openai_finetune(self):
        m = lazyllm.OnlineChatModule(source="openai", model="gpt-3.5-turbo-0125", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_openai_isinstance(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=True)
        self.assertTrue(isinstance(m, lazyllm.OnlineChatModule))

    def test_openai_function_call(self):
        m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=False, prompter=prompter)

        query = "What's the weather like today in Tokyo"
        history = []
        t = json.dumps(tools)
        input = {"tools": t, "input": query}
        resp = m(input, llm_chat_history=history, tools=tools)
        lazyllm.LOG.info(resp)
        if resp.get("finish_reason", "") == "tool_calls":
            resp = resp['message']
            history.append({"role": "user", "content": query})
            history.append(resp)
            tool_calls = resp["tool_calls"]
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                if tool_name == "get_current_weather":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    output = get_current_weather(location, unit)
                elif tool_name == "get_n_day_weather_forecast":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    n_days = tool_args["n_days"]
                    output = get_n_day_weather_forecast(location, unit, n_days)
                else:
                    output = ""

                history.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": output})
                resp = m("", llm_chat_history=history, tools=tools)
                lazyllm.LOG.info(resp)

    def test_kimi_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="kimi", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                    delta = r["choices"][0]["delta"]
                    if "content" in delta:
                        content += delta["content"]
                print(f"response: {content}")
            history.append([query, content])
            print(f"history: {history}")

    def test_kimi_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="kimi", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            resp = resp['message']
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
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                    delta = r["choices"][0]["delta"]
                    content += delta["content"]
                print(f"response: {content}")
            history.append([query, content])
            print(f"history: {history}")

    def test_glm_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="glm", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            resp = resp['message']
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_glm_finetune(self):
        m = lazyllm.OnlineChatModule(source="glm", model="chatglm3-6b", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_glm_function_call(self):
        m = lazyllm.OnlineChatModule(source="glm", stream=False, prompter=prompter)

        query = "What's the weather like today in Tokyo"
        history = []
        t = json.dumps(tools)
        input = {"tools": t, "input": query}
        resp = m(input, llm_chat_history=history, tools=tools)
        lazyllm.LOG.info(resp)
        if resp.get("finish_reason", "") == "tool_calls":
            resp = resp['message']
            history.append({"role": "user", "content": query})
            history.append(resp)
            tool_calls = resp["tool_calls"]
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                if tool_name == "get_current_weather":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    output = get_current_weather(location, unit)
                elif tool_name == "get_n_day_weather_forecast":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    n_days = tool_args["n_days"]
                    output = get_n_day_weather_forecast(location, unit, n_days)
                else:
                    output = ""

                history.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": output})
                resp = m("", llm_chat_history=history, tools=tools)
                lazyllm.LOG.info(resp)

    def test_qwen_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="qwen", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                    delta = r["choices"][0]["delta"]
                    content += delta["content"]
                print(f"response: {content}")
            history.append([query, content])
            print(f"history: {history}")

    def test_qwen_nonstream_inference(self):
        m = lazyllm.OnlineChatModule(source="qwen", stream=False)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        for query in querys:
            resp = m(query, llm_chat_history=history)
            resp = resp['message']
            history.append([query, resp['content']])
            print(f"history: {history}")

    def test_qwen_finetune(self):
        m = lazyllm.OnlineChatModule(source="qwen", model="qwen-turbo", stream=True)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file)
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_qwen_function_call(self):
        m = lazyllm.OnlineChatModule(source="qwen", stream=False, prompter=prompter)

        query = "What's the weather like today in Tokyo"
        history = []
        t = json.dumps(tools)
        input = {"tools": t, "input": query}
        resp = m(input, llm_chat_history=history, tools=tools)
        lazyllm.LOG.info(resp)
        if resp.get("finish_reason", "") == "tool_calls":
            resp = resp['message']
            history.append({"role": "user", "content": query})
            history.append(resp)
            tool_calls = resp["tool_calls"]
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                if tool_name == "get_current_weather":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    output = get_current_weather(location, unit)
                elif tool_name == "get_n_day_weather_forecast":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    n_days = tool_args["n_days"]
                    output = get_n_day_weather_forecast(location, unit, n_days)
                else:
                    output = ""

                history.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": output})
                resp = m("", llm_chat_history=history, tools=tools)
                lazyllm.LOG.info(resp)

    def test_sensenova_stream_inference(self):
        m = lazyllm.OnlineChatModule(source="sensenova", stream=True)
        querys = ['Hello!', '你是谁', '你会做什么', '讲个笑话吧', '你懂摄影吗']
        history = []
        import time
        for query in querys:
            time.sleep(2)
            resp = m(query, llm_chat_history=history)
            content = ""
            for r in resp:
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                    delta = r["choices"][0]["delta"]
                    content += delta["content"]
                print(f"response: {content}")
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
            history.append([query, resp["message"]])
            print(f"history: {history}")

    def test_sensenova_finetune(self):
        m = lazyllm.OnlineChatModule(source="sensenova", model="nova-ptc-s-v2", stream=False)
        train_file = "<trainging file>"
        m.set_train_tasks(train_file=train_file, upload_url="https://file.sensenova.cn/v1/files")
        m._get_train_tasks()
        m._get_deploy_tasks()

    def test_sensenova_function_call(self):
        m = lazyllm.OnlineChatModule(source="sensenova", stream=False, prompter=prompter)

        query = "What's the weather like today in Tokyo"
        history = []
        t = json.dumps(tools)
        input = {"tools": t, "input": query}
        resp = m(input, llm_chat_history=history, tools=tools)
        lazyllm.LOG.info(resp)
        if resp.get("finish_reason", "") == "tool_calls":
            history.append({"role": "user", "content": query})
            history.append(resp)
            tool_calls = resp["tool_calls"]
            for tool_call in tool_calls:
                tool_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                if tool_name == "get_current_weather":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    output = get_current_weather(location, unit)
                elif tool_name == "get_n_day_weather_forecast":
                    tool_args = tool_call["function"]["arguments"]
                    tool_args = json.loads(tool_args)
                    location = tool_args["location"]
                    unit = tool_args["unit"]
                    n_days = tool_args["n_days"]
                    output = get_n_day_weather_forecast(location, unit, n_days)
                else:
                    output = ""

                history.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": output})
                resp = m("", llm_chat_history=history, tools=tools)
                lazyllm.LOG.info(resp)


if __name__ == '__main__':
    unittest.main()

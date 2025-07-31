# 自适应聊天机器人构建教程
本教程展示了如何使用 LazyLLM 构建一个交互式聊天机器人，基于 SenseNova 模型实现自然语言处理。您将学习如何初始化模型、配置环境变量以及实现一个简单的用户交互循环。

通过本教程您将学习以下内容

- 如何通过 [OnlineChatModule][lazyllm.OnlineChatModule] 初始化 SenseNova 模型。
- 如何设置 API Key 和 Secret Key 以正确连接 SenseNova。
- 如何编写一个基于用户输入与模型响应的交互式聊天机器人。
# 环境准备
在开始之前，请确保已安装 LazyLLM：

```bash
pip install lazyllm
```
# 代码实现

## Step 1: 配置环境变量
为了使用 SenseNova 在线模型，需要设置 API Key 和 Secret Key。这些密钥可以通过环境变量的方式配置。

```python
import os

# 设置 LazyLLM 的 SenseNova API Key 和 Secret Key
os.environ["LAZYLLM_SENSENOVA_API_KEY"] = "YOUR_API_KEY"
os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = "YOUR_SECRET_KEY"
```
参数说明：
LAZYLLM_SENSENOVA_API_KEY：SenseNova 提供的 API Key，用于身份验证。
LAZYLLM_SENSENOVA_SECRET_KEY：SenseNova 提供的 Secret Key，用于安全验证。
确保将 "YOUR_API_KEY" 和 "YOUR_SECRET_KEY" 替换为您从 SenseNova 获取的实际密钥。

## Step 2: 初始化 OnlineChatModule
LazyLLM 提供了 OnlineChatModule 用于连接在线模型（如 SenseNova）。我们可以通过以下代码初始化模块：

```python

import lazyllm

# 初始化 LazyLLM 的在线聊天模块 (基于 SenseNova)
model = lazyllm.OnlineChatModule(provider="sensenova")
```
参数说明：
provider="sensenova"：指定使用 SenseNova 模型作为聊天服务提供方。
## Step 3: 构建用户交互循环
通过一个简单的 while 循环，可以实现用户与模型的实时交互：

```python

if __name__ == "__main__":
    print("Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # 使用 LazyLLM 模型生成回复
        response = model(user_input)
        print(f"Model: {response}")
```
实现功能：
用户输入会传递给 SenseNova 模型进行处理。
模型返回的响应会打印到控制台。
输入 "exit" 或 "quit" 可退出对话。
### 完整代码
以下是完整的代码实现：

```python

import os
import lazyllm

# 设置 LazyLLM 的 SenseNova API Key 和 Secret Key
os.environ["LAZYLLM_SENSENOVA_API_KEY"] = "YOUR_API_KEY"
os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = "YOUR_SECRET_KEY"

# 初始化 LazyLLM 的在线聊天模块 (基于 SenseNova)
model = lazyllm.OnlineChatModule(provider="sensenova")

# 测试模型交互
if __name__ == "__main__":
    print("Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # 使用 LazyLLM 模型生成回复
        response = model(user_input)
        print(f"Model: {response}")
```
### 示例运行结果
启动聊天机器人
运行脚本后，您会看到以下提示：

```bash
Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.
You:
```
示例对话
```bash
You: Hello!
Model: Hello! How can I assist you today?

You: What's the weather like today?
Model:  I don't have real-time capabilities or access to current data, including weather updates. To find out the weather for today, I recommend checking a reliable weather website like the Weather Channel, BBC Weather, or using a weather app on your smartphone for the most accurate and up-to-date information.

You: exit
Exiting...
```
小贴士
环境变量配置：

如果不想在代码中硬编码 API Key 和 Secret Key，可以通过 .env 文件或系统配置管理环境变量。
错误处理：

如果模型未能返回响应，请确保 API Key 和 Secret Key 配置正确。
检查网络连接是否稳定。
扩展功能：

您可以集成更多 LazyLLM 的功能（如工具函数或 RAG 推理）来增强聊天机器人的能力。

## 结语
通过本教程，您已经学会如何使用 LazyLLM 和 SenseNova 构建一个简单的聊天机器人。该机器人可以作为基础，进一步扩展为多功能的 AI 助手。尝试添加更多功能或结合工具函数来提高机器人的交互能力吧！


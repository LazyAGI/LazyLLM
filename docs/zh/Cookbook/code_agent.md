# 代码助手代理(CodeAssistantAgent)

本项目展示了一个基于大语言模型的智能代码助手，可识别用户意图并执行代码生成、代码解释和思路总结等功能。

!!! abstract "通过本节您将学习到以下内容:"

    - 如何构建多功能的代码助手代理
    - 如何实现意图识别与任务分发
    - 如何实现交互式的代码生成与解释功能

## 项目依赖

确保安装以下依赖：

```bash
pip install lazyllm
```

导入相关包：

```python
from lazyllm.tools import CodeGenerator, IntentClassifier
from lazyllm import OnlineChatModule
```

## 代码实现

``` python
from typing import List, Dict, Any
import lazyllm
## Step 1: 初始化代理

**功能说明：**
- 配置大语言模型和意图分类器
- 设置默认意图列表和示例

**参数说明**
- llm (str): 大语言模型实例
- intent_list (List[str]): 支持的意图列表，默认为['生成代码', '解释', '总结']
- prompt (str): 基础提示词
- intent_constrain (str): 意图分类约束条件
- intent_attention (str): 意图分类注意事项
- intent_examples (List[List[str]]): 意图分类示例
- return_trace (bool): 是否返回执行轨迹，默认为False
class CodeAssistantAgent:
    def __init__(self, llm: str,
                 intent_list: List[str] = None,
                 prompt: str = '',
                 intent_constrain: str = '',
                 intent_attention: str = '',
                 intent_examples: List[List[str]] = None,
                 return_trace: bool = False):
        self.generator = CodeGenerator(base_model=llm, prompt=prompt)
        self.intent_classifier = IntentClassifier(
            llm=llm,
            intent_list=intent_list or ['生成代码', '解释', '总结'],
            prompt=prompt,
            constrain=intent_constrain,
            attention=intent_attention,
            examples=intent_examples or [],
            return_trace=return_trace
        )
## Step 2: 代码生成
**功能说明：**
- 根据指令生成Python代码
    def generate_code(self, instruction: str, context: str = "") -> str:
        prompt = f"{context}\n\n请根据以下要求编写Python代码:\n{instruction}" if context else f"请根据以下要求编写Python代码:\n{instruction}"
        return self.generator(prompt)
## Step 3: 代码解释
**功能说明：**
- 解释代码逻辑和功能
    def explain_code(self, code: str) -> str:
        prompt = f"请为以下Python代码添加详细注释并解释其逻辑：\n{code}"
        return self.generator(prompt)
## Step 4: 对话总结
**功能说明：**
- 生成结构化摘要
    def summarize_thoughts(self, history: List[Dict[str, Any]]) -> str:
        convo = "\n".join([f"用户: {m['user']}\n助手: {m['assistant']}" for m in history])
        prompt = f"请基于以下对话内容，总结出核心思路和流程：\n{convo}"
        return self.generator(prompt)
## Step 5: 交互控制
**功能说明：**
- 管理对话循环, 维护对话历史
    def interactive_mode(self, context: str = "", history: List[Dict[str, Any]] = None):
        if history is None:
            history = []
        print("进入交互模式，输入'exit'退出...")
        while True:
            user_input = input("\n您: ")
            if user_input.lower() == 'exit':
                break
                
            intent = self.intent_classifier.forward(user_input, llm_chat_history=history)
            try:
                if intent == '生成代码':
                    result = self.generate_code(user_input, context)
                    print("\n[代码生成] 生成的代码如下：\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                elif intent == '解释':
                    last_code = history[-1]['assistant'] if history else ''
                    result = self.explain_code(last_code)
                    print("\n[代码解释] 解释结果：\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                elif intent == '总结':
                    result = self.summarize_thoughts(history)
                    print("\n[思路总结] 总结如下：\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                else:
                    print("无法识别您的需求，默认进行代码生成。")
                    result = self.generate_code(user_input, context)
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

            except Exception as e:
                print(f"执行时出错: {e}")



```

## 示例运行结果

#### 示例场景：
```python
if __name__ == '__main__':
    chat = lazyllm.OnlineChatModule()
    assistant = CodeAssistantAgent(
        llm=chat,
        intent_list=['生成代码', '解释', '总结'],
        intent_examples=[
            ['请实现排序函数', '生成代码'],
            ['写一个代码来实现agent流程', '生成代码'],
            ['解释代码意义功能', '解释'],
            ['这段代码什么意思？', '解释'],
            ['现在能给出思路总结吗', '总结']
        ]
    )
    assistant.interactive_mode()
```

**输入**
"生成一段二分类算法的代码"

**程序控制台输出：**
```python
[代码生成] 生成的代码如下：


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def binary_classification(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Perform binary classification using Logistic Regression.

    Args:
        data (np.ndarray): The input features, a 2D numpy array where each row represents a sample.
        labels (np.ndarray): The labels for each sample, a 1D numpy array containing binary labels (0 or 1).

    Returns:
        float: The accuracy of the model on the test set.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    predictions = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

# Example usage:
if __name__ == "__main__":
    # Example data: 4 samples with 2 features each
    data = np.array([[0.5, 1.2], [1.3, 3.4], [3.5, 2.1], [2.2, 4.3]])
    labels = np.array([0, 0, 1, 1])

    # Perform binary classification and print the accuracy
    acc = binary_classification(data, labels)
    print(f"Model Accuracy: {acc:.2f}")
```
**输入**
"解释这代码"

**程序控制台输出：**
```python
[代码解释] 解释结果：


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def binary_classification(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Perform binary classification using Logistic Regression.

    Args:
        data (np.ndarray): The input features, a 2D numpy array where each row represents a sample.
        labels (np.ndarray): The labels for each sample, a 1D numpy array containing binary labels (0 or 1).

    Returns:
        float: The accuracy of the model on the test set.
    """
    # Split the data into training and testing sets
    # This function divides the data into training and testing subsets.
    # 'test_size=0.2' indicates that 20% of the data is used for testing, and the rest for training.
    # 'random_state=42' ensures that the split is reproducible.
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    # LogisticRegression is a classifier that uses logistic functions to model the probability of a binary outcome.
    model = LogisticRegression()

    # Train the model
    # The fit method trains the model using the training data and corresponding labels.
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    # The predict method outputs the predicted labels for the test set.
    predictions = model.predict(X_test)

    # Calculate the accuracy of the model
    # accuracy_score computes the accuracy, a performance metric for classification models, as the number of correct predictions divided by the total number of predictions.
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

# Example usage:
if __name__ == "__main__":
    # Example data: 4 samples with 2 features each
    # This is a small dataset with 4 samples, each having 2 features.
    data = np.array([[0.5, 1.2], [1.3, 3.4], [3.5, 2.1], [2.2, 4.3]])
    labels = np.array([0, 0, 1, 1])

    # Perform binary classification and print the accuracy
    # The binary_classification function is called with example data and labels, and the resulting accuracy is printed.
    acc = binary_classification(data, labels)
    print(f"Model Accuracy: {acc:.2f}")
```
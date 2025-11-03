# Code Assistant Agent (CodeAssistantAgent)

## Project Overview

This project demonstrates an intelligent code assistant powered by a large language model.  
It can identify user intent and perform tasks such as code generation, code explanation, and idea summarization.

!!! abstract "In this section, you will learn the following:"

    - How to build a multi-functional code assistant agent  
    - How to implement intent recognition and task dispatching  
    - How to create interactive features for code generation and explanation  

## Project Dependencies

Make sure the following dependency is installed:

```bash
pip install lazyllm
```
## Code Implementation
``` python
from typing import List, Dict, Any
import lazyllm
## Step 1: Initialize the Agent

**Function Description：**
- Configure the large language model and intent classifier
- Set default intent list and examples

**Parameter Description:**
- llm (str): The large language model instance
- intent_list (List[str]): Supported intents, default is ['Code Generation', 'Explanation', 'Summarization']
- prompt (str): Base prompt for the model
- intent_constrain (str): Constraint conditions for intent classification
- intent_attention (str): Notes or special considerations for intent classification
- intent_examples (List[List[str]]): Example pairs for intent classification
- return_trace (bool): Whether to return execution trace, default is False
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
            intent_list = intent_list or ['code generation', 'explanation', 'summarization']
            prompt=prompt,
            constrain=intent_constrain,
            attention=intent_attention,
            examples=intent_examples or [],
            return_trace=return_trace
        )
## Step 2: Code Generation  
**Function Description:**  
- Generate Python code based on the given instruction  
    def generate_code(self, instruction: str, context: str = "") -> str:
        prompt = f"{context}\n\nPlease write Python code according to the following instruction:\n{instruction}" if context else f"Please write Python code according to the following instruction:\n{instruction}"
        return self.generator(prompt)
## Step 3: Code Explanation
**Function Description：**
- Explain the logic and function of the given code
    def explain_code(self, code: str) -> str:
        prompt = f"Please add detailed comments and explain the logic of the following Python code:\n{code}"
        return self.generator(prompt)

## Step 4: Explain the logic and function of the given code
**Function Description：**
- Generate a structured summary of the conversation
    def summarize_thoughts(self, history: List[Dict[str, Any]]) -> str:
        convo = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history])
        prompt = f"Based on the following conversation, summarize the core ideas and workflow:\n{convo}"
        return self.generator(prompt)

## Step 5: Interactive Control
**Function Description：**
- Manage the interactive loop and maintain conversation history
    def interactive_mode(self, context: str = "", history: List[Dict[str, Any]] = None):
        if history is None:
            history = []
        print("Entering interactive mode. Type 'exit' to quit...")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break

            intent = self.intent_classifier.forward(user_input, llm_chat_history=history)
            try:
                if intent == 'code generation':
                    result = self.generate_code(user_input, context)
                    print("\n[Code Generation] Generated code:\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                elif intent == 'explanation':
                    last_code = history[-1]['assistant'] if history else ''
                    result = self.explain_code(last_code)
                    print("\n[Code Explanation] Explanation result:\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                elif intent == 'summarization':
                    result = self.summarize_thoughts(history)
                    print("\n[Thought Summarization] Summary:\n")
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

                else:
                    print("Unable to recognize your intent, defaulting to code generation.")
                    result = self.generate_code(user_input, context)
                    print(result)
                    history.append({'user': user_input, 'assistant': result})

            except Exception as e:
                print(f"Error during execution: {e}")

```

## Example Execution

#### Example Scenario:

```python
if __name__ == '__main__':
    chat = lazyllm.OnlineChatModule()
    assistant = CodeAssistantAgent(
        llm=chat,
        intent_list=['code generation', 'explanation', 'summarization'],
        intent_examples=[
            ['Please implement a sorting function', 'code generation'],
            ['Write code to implement an agent flow', 'code generation'],
            ['Explain what this code does', 'explanation'],
            ['What does this code mean?', 'explanation'],
            ['Can you now provide a summary of the thought process?', 'summarization']
        ]
    )
    assistant.interactive_mode()


**Input**  
"Generate a binary classification algorithm"

**Console Output:**
```python
[Code Generation] The generated code is as follows:



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
**Input**  
"Explain this code"

**Console Output:**
```python
[Code Explanation] Explanation result:



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
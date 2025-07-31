# Adaptive Chatbot Construction Tutorial
This tutorial demonstrates how to build an interactive chatbot using LazyLLM, leveraging the SenseNova model for natural language processing.You will learn how to initialize the model, configure environment variables, and implement a simple user interaction loop.

By completing this tutorial, you will learn the following:

- How to initialize the SenseNova model using the [OnlineChatModule][lazyllm.OnlineChatModule].
- How to set up the API Key and Secret Key to connect to SenseNova correctly.
- How to build an interactive chatbot that responds to user input.
# Environment Setup
Before starting, ensure you have installed LazyLLM:

```bash
pip install lazyllm
```
# Code Implementation

## Step 1: Configure Environment Variables
To use the SenseNova online model, you need to set the API Key and Secret Key. These credentials can be configured as environment variables.

```python
import os

# Configure LazyLLM with SenseNova API Key and Secret Key
os.environ["LAZYLLM_SENSENOVA_API_KEY"] = "YOUR_API_KEY"
os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = "YOUR_SECRET_KEY"
```
Parameter Description:
LAZYLLM_SENSENOVA_API_KEY: The API Key provided by SenseNova for authentication.
LAZYLLM_SENSENOVA_SECRET_KEY: The Secret Key provided by SenseNova for secure access.
Make sure to replace "YOUR_API_KEY" and "YOUR_SECRET_KEY" with the actual credentials you received from SenseNova.

## Step 2: Initialize the OnlineChatModule
LazyLLM provides the OnlineChatModule to connect to online models like SenseNova. You can initialize the module using the following code:

```python

import lazyllm

# Initialize LazyLLM's online chat module (using SenseNova)
model = lazyllm.OnlineChatModule(provider="sensenova")
```
Parameter Description:
provider="sensenova": Specifies that the SenseNova model is being used as the chat service provider.
## Step 3: Build a User Interaction Loop
Using a simple while loop, you can create real-time interactions between the user and the model:

```python

if __name__ == "__main__":
    print("Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Generate a response using the LazyLLM model
        response = model(user_input)
        print(f"Model: {response}")
```
Features Implemented:
User input is sent to the SenseNova model for processing.
The model's response is printed to the console.
Typing "exit" or "quit" ends the conversation.
### Complete Code
Below is the complete implementation:

```python

import os
import lazyllm

# Configure LazyLLM with SenseNova API Key and Secret Key
os.environ["LAZYLLM_SENSENOVA_API_KEY"] = "YOUR_API_KEY"
os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = "YOUR_SECRET_KEY"

# Initialize LazyLLM's online chat module (using SenseNova)
model = lazyllm.OnlineChatModule(provider="sensenova")

# Test the chatbot
if __name__ == "__main__":
    print("Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Generate a response using the LazyLLM model
        response = model(user_input)
        print(f"Model: {response}")
```
### Example Run
Start the Chatbot
When you run the script, you will see the following prompt:

```bash
Welcome to LazyLLM Chatbot! Type 'exit' or 'quit' to end the chat.
You:
```
Sample Conversation
```bash
You: Hello!
Model: Hello! How can I assist you today?

You: What's the weather like today?
Model:  I don't have real-time capabilities or access to current data, including weather updates. To find out the weather for today, I recommend checking a reliable weather website like the Weather Channel, BBC Weather, or using a weather app on your smartphone for the most accurate and up-to-date information.

You: exit
Exiting...
```
Tips
Environment Variable Configuration
If you prefer not to hardcode the API Key and Secret Key in your script, you can use a .env file or system configuration to manage environment variables.
Error Handling
If the model fails to return a response, ensure that the API Key and Secret Key are configured correctly.
Confirm that your network connection is stable.
Extending Functionality
You can integrate additional LazyLLM features (e.g., tool functions or RAG reasoning) to enhance the chatbot's capabilities.

## Conclusion
By following this tutorial, you have learned how to build a simple chatbot using LazyLLM and SenseNova. This chatbot serves as a foundation that can be expanded into a more versatile AI assistant. Try adding more features or combining tool functions to improve the chatbot's interactivity!


import lazyllm
from lazyllm.tools.memory import Memory
import os
import logging

# Set logging level to CRITICAL to suppress INFO/ERROR logs
logging.getLogger("powermem").setLevel(logging.CRITICAL)
logging.getLogger("audit").setLevel(logging.CRITICAL)

os.environ['LAZYLLM_QWEN_API_KEY'] = 'yourapikey'
chat = lazyllm.OnlineChatModule()
user_id = 'test_user'
memory = Memory(source='powermem')

history = []

while True:
    query = input("query(enter 'quit' to exit): ")
    if query == "quit":
        break
    # Retrieve relevant context from memory
    retrieved_context = memory.get(query=query, user_id=user_id)

    if retrieved_context and len(retrieved_context.strip()) > 0:
        print(f"  └─ [Memory Triggered]: Found information...")
        # Append memory to the beginning of the prompt as background knowledge
        full_prompt = (
            f"Background Info:\n{retrieved_context}\n\n"
            f"User Query: {query}\n"
            f"Please answer the user query based on the background info."
        )
    else:
        full_prompt = query

    res = chat(full_prompt, llm_chat_history=history)
    print(f"AI  : {str(res)}")

    # Store current conversation (original query + response) into PowerMem
    memory_content = f"User: {query}\nAI: {str(res)}"
    memory.add(memory_content, user_id=user_id)

    # Update short-term history
    history.append([query, res])

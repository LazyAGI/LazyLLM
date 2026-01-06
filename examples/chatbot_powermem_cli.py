import lazyllm
from lazyllm.tools.memory import Memory
from lazyllm.tools.sql import SqlManager
import os
import logging

# Set logging level to CRITICAL to suppress INFO/ERROR logs
logging.getLogger("powermem").setLevel(logging.CRITICAL)
logging.getLogger("audit").setLevel(logging.CRITICAL)

os.environ['LAZYLLM_QWEN_API_KEY'] = 'yourapikey'
chat = lazyllm.OnlineChatModule()
user_id = 'test_user'
memory = Memory(source='powermem')
# default sqlite db path of powermem
db_path = './data/powermem_dev.db'
# Initialize SqlManager pointing to the same SQLite file
sql_tool = SqlManager(
    db_type='sqlite',
    user='',
    password='',
    host='',
    port=0,
    db_name=db_path
)
print(f"SqlManager initialized. Found tables: {sql_tool.get_all_tables()}")
history = []

print("-" * 50)
print("Chatbot started. Special commands:")
print("  quit         : Exit the program")
print("  sql: <query> : Execute raw SQL directly (e.g., 'sql: select count(*) from memories')")
print("-" * 50)

while True:
    query = input("\nquery: ")

    if query == "quit":
        break

    # --- SQL Mode: Direct database query via SqlManager ---
    if query.lower().startswith("sql:"):
        sql_statement = query[4:].strip()
        print(f"Executing SQL: {sql_statement}")
        try:
            # Use SqlManager to execute the query
            result = sql_tool.execute_query(sql_statement)
            print(f"SQL Result:\n{result}")
        except Exception as e:
            print(f"SQL Error: {e}")
        continue

    # --- Chat Mode: Standard RAG flow ---

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

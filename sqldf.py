import sqlite3
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser

# ✅ Step 1: Setup Databases
meeting_db = SQLDatabase.from_uri("sqlite:///meeting_db.sqlite")  # Meeting database
chat_db_conn = sqlite3.connect("chat_history.sqlite")  # Chat history database
cursor = chat_db_conn.cursor()

# ✅ Step 2: Create Chat History Table (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    bot TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")
chat_db_conn.commit()


# ✅ Step 3: Function to Insert Chat Messages
def save_chat(user, bot):
    """Stores user questions & bot responses in chat history."""
    conn = sqlite3.connect("chat_history.sqlite")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user, bot, timestamp) VALUES (?, ?, datetime('now'))",
                   (user, bot))
    conn.commit()
    conn.close()


# ✅ Step 4: Retrieve Last N Chat Messages as Context
def get_chat_history(limit=5):
    """Fetches the last N user-bot interactions for context."""
    conn = sqlite3.connect("chat_history.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT user, bot FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
    chat_history = cursor.fetchall()
    conn.close()

    # Format chat history as a readable string
    return "\n".join([f"User: {user}\nBot: {bot}" for user, bot in reversed(chat_history)])


# ✅ Step 5: Modify `extract_sql_query` to Use Chat History
def extract_sql_query(inputs):
    """Generates an SQL query dynamically, considering previous context."""
    user_input = inputs["input"]
    chat_history = inputs["chat_history"]

    # Example Rule: If "above project" is mentioned, extract project name from chat history
    if "above project" in user_input.lower():
        last_project_query = None
        for line in chat_history.split("\n"):
            if "project" in line.lower():
                last_project_query = line.split(":")[-1].strip()
                break

        if last_project_query:
            user_input = user_input.replace("above project", last_project_query)

    # Example SQL generation rule
    if "key points" in user_input.lower():
        return f"SELECT key_points FROM meetings WHERE project_name LIKE '%{user_input.split()[-1]}%' ORDER BY date DESC LIMIT 1;"

    return "SELECT * FROM meetings LIMIT 5;"  # Default query


# ✅ Step 6: Function to Execute SQL Query
def execute_query(query):
    """Executes a SQL query and fetches results."""
    with sqlite3.connect("meeting_db.sqlite") as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
    return result


# ✅ Step 7: Initialize LLM (GPT-4)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ✅ Step 8: Define the Chain
chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(lambda _: get_chat_history(5))  # Get last 5 interactions
).assign(
    query=RunnableLambda(extract_sql_query)  # Generate SQL Query based on chat history
).assign(
    result=itemgetter("query") | RunnableLambda(execute_query)  # Execute SQL Query
).assign(
    llm_input=lambda
        inputs: f"Chat History:\n{inputs['chat_history']}\n\nUser Question: {inputs['input']}\n\nSQL Query Result: {inputs['result']}"
    # Combine everything
).assign(
    final_response=itemgetter("llm_input") | StrOutputParser() | llm  # Process with LLM
).assign(
    log_chat=lambda inputs: (save_chat(inputs["input"], inputs["final_response"]))  # Store chat history
)

# ✅ Step 9: Run the Chain
user_question = "What are the key points of the above project?"
response = chain.invoke({"input": user_question})

print("Bot:", response["final_response"])  # LLM Answer

import sqlite3
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import create_sql_query_chain
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


# ✅ Step 5: Initialize LLM (GPT-4)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ✅ Step 6: Create SQL Query Chain
sql_chain = create_sql_query_chain(llm, meeting_db)


# ✅ Step 7: Function to Generate & Validate SQL Query in a Loop
def generate_valid_sql_query(user_question, chat_history, max_attempts=5):
    """Generates and refines an SQL query up to `max_attempts` times until it's correct."""

    for attempt in range(1, max_attempts + 1):
        print(f"🔄 Attempt {attempt}: Generating SQL Query...")

        # Step 1: Generate SQL Query
        prompt = f"""
        You are an expert SQL generator for a meeting database. 
        Your task is to generate the most accurate SQL query based on the user's question and chat history.

        - **Chat History (for context):**  
        {chat_history}

        - **User Question:**  
        {user_question}

        **Instructions:**  
        1. Ensure the query retrieves relevant meeting details.  
        2. If the user references a previous project or topic, infer it from the chat history.  
        3. Optimize the query for efficiency.  
        4. Return **only the SQL query** with no explanation.  

        Now, generate the correct SQL query:
        """
        sql_query = sql_chain.invoke(prompt).strip()

        # Step 2: Ask LLM to Validate the Query
        validation_prompt = f"""
        You are an expert SQL validator. Given the SQL query below, check if it correctly answers the user's question.

        - **SQL Query:**  
        {sql_query}

        - **User Question:**  
        {user_question}

        **Validation Rules:**  
        1. Does the query retrieve relevant data?  
        2. Does it correctly reference previous topics if needed?  
        3. Is it free of syntax errors?  
        4. If incorrect, suggest an improved query.  

        If correct, return **"VALID"**. Otherwise, return a corrected SQL query.
        """
        validation_response = llm.invoke(validation_prompt).strip()

        # Step 3: If Valid, Return the Query; Otherwise, Retry
        if validation_response == "VALID":
            print(f"✅ Attempt {attempt}: Query validated successfully!")
            return sql_query
        else:
            print(f"⚠️ Attempt {attempt}: Query corrected, retrying...")
            sql_query = validation_response  # Use the corrected query

    print("❌ Maximum validation attempts reached. Using last refined SQL query.")
    return sql_query  # Return the last refined query even if not marked "VALID"


# ✅ Step 8: Define the Runnable Chain
chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(lambda _: get_chat_history(5))  # Retrieve last 5 chat messages
).assign(
    query=lambda inputs: generate_valid_sql_query(inputs["input"], inputs["chat_history"])
    # Generate & validate SQL Query
).assign(
    result=itemgetter("query") | RunnableLambda(lambda q: meeting_db.run(q))  # Execute SQL Query
).assign(
    llm_input=lambda
        inputs: f"Chat History:\n{inputs['chat_history']}\n\nUser Question: {inputs['input']}\n\nSQL Query Result: {inputs['result']}"
    # Combine everything
).assign(
    final_response=itemgetter("llm_input") | StrOutputParser() | llm  # Pass to LLM for response
).assign(
    log_chat=lambda inputs: (save_chat(inputs["input"], inputs["final_response"]))  # Store in chat history
)

# ✅ Step 9: Run the Chain
user_question = "What were the key discussion points in the last meeting?"
response = chain.invoke({"input": user_question})

print("Bot:", response["final_response"])  # LLM Answer

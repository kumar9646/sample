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


# ✅ Step 7: Function to Validate and Refine SQL Query
def validate_and_refine_sql(user_question, chat_history, max_attempts=5):
    """Refines the SQL query up to `max_attempts` times until it's valid."""
    for attempt in range(max_attempts):
        # Step 1: Generate SQL Query
        sql_query = sql_chain.invoke({"question": user_question, "chat_history": chat_history})

        # Step 2: Ask LLM if the SQL Query is Correct
        validation_prompt = f"""
        You are an expert SQL validator. Given the following query:
        {sql_query}

        - Check if the query is correct and relevant to the question: "{user_question}".
        - Ensure it retrieves the right information from the database.
        - If incorrect, suggest improvements.

        Return "VALID" if correct, or a corrected query.
        """
        validation_response = llm.invoke(validation_prompt).strip()

        # Step 3: If LLM says "VALID", use the query; otherwise, refine it
        if validation_response == "VALID":
            return sql_query  # Use the validated query
        else:
            print(f"Attempt {attempt + 1}: Refining SQL Query...")
            sql_query = validation_response  # Update query with LLM's correction

    print("Maximum validation attempts reached. Using last refined SQL query.")
    return sql_query  # Return the last refined query even if not marked "VALID"


# ✅ Step 8: Define the Runnable Chain with Validation Loop
chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(lambda _: get_chat_history(5))  # Retrieve last 5 chat messages
).assign(
    query=lambda inputs: validate_and_refine_sql(inputs["input"], inputs["chat_history"])  # Generate & refine SQL Query
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

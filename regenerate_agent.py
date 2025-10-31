from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool

# -----------------------------
# Step 1. Prepare Chat History
# -----------------------------
chat_history = [
    {"role": "human", "content": "in which meeting gao lee and jang participated"},
    {"role": "AI", "content": "following are the list of meetings: x1 meeting, x2 meetings and x3 meetings"},
    {"role": "user", "content": "can you summarize the project discussed in the above meetings"}
]

# Convert to readable text for prompt
chat_history_text = "\n".join(
    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[:-1]]
)
latest_query = chat_history[-1]["content"]

# -----------------------------
# Step 2. Define Reformulation Chain
# -----------------------------
reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=(
        "Given the following conversation history:\n"
        "{chat_history}\n\n"
        "Reformulate the user's latest query so it is self-contained, clear, "
        "and contextually complete. Return only the reformulated query.\n\n"
        "User query: {user_query}\n\nReformulated query:"
    ),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

reformulate_chain = LLMChain(
    llm=llm,
    prompt=reformulate_prompt
)

# -----------------------------
# Step 3. Define the Tools
# -----------------------------

# 1️⃣ Reformulator tool
def reformulate_tool_func(query: str):
    return reformulate_chain.run({
        "chat_history": chat_history_text,
        "user_query": query
    })

reformulate_tool = Tool(
    name="QueryReformulator",
    func=reformulate_tool_func,
    description="Reformulates a user query using chat history to make it self-contained."
)

# 2️⃣ Retriever tool (you can replace with a real vector retriever)
def retrieve_documents(query: str):
    return f"Retrieved docs related to: '{query}'"

retriever_tool = Tool(
    name="Retriever",
    func=retrieve_documents,
    description="Retrieves meeting or project-related documents for a given reformulated query."
)

# -----------------------------
# Step 4. Initialize the Agent
# -----------------------------
agent = initialize_agent(
    tools=[reformulate_tool, retriever_tool],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

# -----------------------------
# Step 5. Run the Agent
# -----------------------------
# The agent can autonomously call tools based on your instruction
query_to_answer = latest_query

agent_prompt = f"""
You are an intelligent meeting assistant.

Before using the Retriever tool, always use the QueryReformulator tool 
to rewrite the user query based on chat history so it’s self-contained.
Then use the Retriever to fetch documents or summaries.

User query: {query_to_answer}
"""

response = agent.run(agent_prompt)
print("\nFinal Answer:\n", response)

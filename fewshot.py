from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
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

reformulate_chain = LLMChain(llm=llm, prompt=reformulate_prompt)

# -----------------------------
# Step 3. Define Tools
# -----------------------------
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

def retrieve_documents(query: str):
    return f"Retrieved docs related to: '{query}'"

retriever_tool = Tool(
    name="Retriever",
    func=retrieve_documents,
    description="Retrieves meeting or project-related documents for a given reformulated query."
)

# -----------------------------
# Step 4. Create Few-Shot Examples
# -----------------------------
examples = [
    {
        "input": (
            "User: What are the topics discussed by Alice and Bob?\n"
            "AI: They attended meetings A1, A2, and A3.\n"
            "User: Can you summarize the discussions?"
        ),
        "output": (
            "First, use the QueryReformulator tool to make the query self-contained. "
            "Reformulated query → 'Summarize the discussions from meetings A1, A2, and A3 attended by Alice and Bob.' "
            "Then use the Retriever tool to fetch relevant documents."
        )
    },
    {
        "input": (
            "User: In which sessions did Maria and John participate?\n"
            "AI: They were part of S1 and S2.\n"
            "User: What were the outcomes of those sessions?"
        ),
        "output": (
            "Use QueryReformulator → 'Summarize the outcomes of sessions S1 and S2 where Maria and John participated.' "
            "Then use the Retriever tool."
        )
    }
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input:\n{input}\nExample Reasoning:\n{output}\n"
)

fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are an intelligent agent that analyzes chat history before answering.",
    suffix=(
        "Conversation:\n{chat_history}\n\n"
        "User's latest query: {user_query}\n\n"
        "Explain what steps you will take and which tools to use (QueryReformulator then Retriever)."
    ),
    input_variables=["chat_history", "user_query"]
)

# -----------------------------
# Step 5. Define LLM with Few-Shot Context
# -----------------------------
fewshot_chain = LLMChain(llm=llm, prompt=fewshot_prompt)

# -----------------------------
# Step 6. Initialize Few-Shot Agent
# -----------------------------
agent = initialize_agent(
    tools=[reformulate_tool, retriever_tool],
    llm=llm,
    agent_type="chat-conversational-react-description",  # Few-shot compatible
    verbose=True
)

# -----------------------------
# Step 7. Run
# -----------------------------
agent_prompt = fewshot_prompt.format(
    chat_history=chat_history_text,
    user_query=latest_query
)

response = agent.run(agent_prompt)
print("\nFinal Answer:\n", response)

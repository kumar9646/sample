from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "user_query"],
    template=(
        "Given the following conversation history:\n{chat_history}\n\n"
        "Reformulate the user's latest query so that it is self-contained, "
        "clear, and contextually complete. Return only the reformulated query.\n\n"
        "User query: {user_query}\n\nReformulated query:"
    ),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reformulate_chain = LLMChain(llm=llm, prompt=reformulate_prompt)



def retrieve_documents(query: str):
    # Normally you'd call retriever.get_relevant_documents(query)
    return f"Retrieved docs for query: '{query}'"

retriever_tool = Tool(
    name="Retriever",
    func=retrieve_documents,
    description="Retrieves relevant meeting or project documents given a reformulated query."
)


def reformulate_query(chat_history: str, user_query: str):
    return reformulate_chain.run({"chat_history": chat_history, "user_query": user_query})

reformulate_tool = Tool(
    name="QueryReformulator",
    func=lambda q: reformulate_query(chat_history, q),
    description="Reformulates a user query using chat history to make it self-contained."
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[reformulate_tool, retriever_tool],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    memory=memory,
    verbose=True
)


chat_history = """User: In which meeting Gao Lee and Jang participated?
AI: Following are the list of meetings: X1 meeting, X2 meeting, and X3 meeting.
"""

user_query = "Can you summarize the project discussed in the above meetings?"

# Reformulate first
reformulated_query = reformulate_query(chat_history, user_query)
print("Reformulated Query:", reformulated_query)

# Now send reformulated query to retriever
result = retrieve_documents(reformulated_query)
print(result)
